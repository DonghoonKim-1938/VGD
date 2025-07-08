import json
import os
from pprint import pprint
import random
import torch
import open_clip
from typing import Dict

from PIL import Image
from diffusers import StableDiffusionPipeline
from torch import nn
from transformers import CLIPModel, AutoModelForCausalLM, AutoTokenizer, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPProcessor
from statistics import mean
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)
from bluestar.utils.chat_template import PromptTemplate as PT
from bluestar.utils.random_utils import set_seed

class VGD(nn.Module):

    def __init__(self, cfg: Dict) -> None :

        super().__init__()
        """Initialize required components."""
        model_ckpt = cfg["model"]["model_ckpt"]

        self.freeze_model = cfg["model"]["freeze_model"]
        self.gen_prompt_only = cfg["model"]["gen_prompt_only"]
        self.get_similarity = cfg["model"]["get_similarity"]

        self.dataset_name = cfg["dataset_name"] if "dataset_name" in cfg.keys() else None

        self.clip_version = cfg["model"]["clip_version"]
        self.clip = CLIPModel.from_pretrained(self.clip_version)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_version)
        self.clip_image_transform = CLIPImageProcessor.from_pretrained(self.clip_version)
        if self.get_similarity:
            self.model_ref, _, self.preprocess_ref = open_clip.create_model_and_transforms(cfg["model"]["clip_ref"],
                                                                                       pretrained=cfg["model"]["clip_ref_pretrain"])
        else:
            self.model_ref = None
            self.preprocess_ref = None

        self.prompt_distillation = cfg["prompt_distillation"] if cfg.get("prompt_distillation") else False
        if self.prompt_distillation:
            self.tokenizer_ref = open_clip.get_tokenizer(cfg["model"]["clip_ref"])

        self.llm = AutoModelForCausalLM.from_pretrained(model_ckpt, load_in_8bit=cfg["model"]["load_in_8bit"])
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)
        self.user_prompt = cfg["model"]["user_prompt"]
        self.system_prompt = cfg["model"]["system_prompt"]
        self.model_prompt = cfg["model"]["model_prompt"]
        self.get_initial_condition = cfg["model"]["get_initial_condition"]

        self.llm_alpha = cfg["model"]["llm_alpha"] if "llm_alpha" in cfg["model"].keys() else 1.0
        self.clip_alpha = cfg["model"]["clip_alpha"] if "clip_alpha" in cfg["model"].keys() else 1.5

        self.beam_expand_factor = cfg["model"]["beam_expand_factor"]
        self.clip_beam_size = cfg["model"]["num_beams"]
        self.llm_beam_size = self.clip_beam_size * self.beam_expand_factor

        self.beam_offset = torch.arange(
            0,
            1 * self.clip_beam_size,
            step=self.clip_beam_size,
            dtype=torch.long,
            device=self.llm.device,
        )

        self.length_cutoff = cfg["length_cutoff"] if cfg.get('length_cutoff') else False
        self.candidate = [[]]
        self.candidate_score = [[]]
        self.done_cnt = 0

        self.batch_size = cfg["batch_size"]

        if self.prompt_distillation:
            self.target_prompt = cfg["target_prompt"] if cfg.get("target_prompt") else None
            distillation_ratio = cfg['distillation_ratio']
            token_org = len(self.clip_tokenizer(self.target_prompt, max_length=77, truncation=True)['input_ids'])-2
            self.max_length = int(token_org * distillation_ratio)
            self.min_length = self.max_length
            self.user_prompt = self.user_prompt.replace('{max_length}', str(self.max_length))
            self.model_prompt = self.model_prompt.replace('{max_length}', str(self.max_length))
        else:
            self.max_length = cfg["model"]["max_length"]
            self.min_length = cfg["model"]["min_length"]

        if not(self.gen_prompt_only):
            self.sd_version = cfg["model"]["sd_version"]
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(self.sd_version) #, torch_dtype=torch.float32)

        self.data_dir = cfg["data_dir"] if cfg.get("data_dir") else None
        self.seed = cfg["seed"]
        self.target_images = [Image.open(image_path) for image_path in self.data_dir] if self.data_dir is not None else None
        self.get_feature()
        self.get_llm_initial_condition()
        self.freeze()

    def get_llm_initial_condition(self):
        if hasattr(self,"all_target_features"):

            self.clip.to("cuda")
            vocab = list(self.clip_tokenizer.get_vocab())
            target_features = self.all_target_features
            image_features = target_features.to("cuda") / target_features.norm(dim=1, keepdim=True).to(
                "cuda")

            if self.get_initial_condition and not self.prompt_distillation:
                best_id = -1
                best_logits = 0

                target_features = self.all_target_features
                for i in range(50):
                    with torch.no_grad():
                        inputs = self.clip_tokenizer(vocab[i * 1000:(i + 1) * 1000], padding=True,
                                                     return_tensors="pt").to("cuda")
                        text_features = self.clip.get_text_features(**inputs).to("cuda")
                        image_features = target_features.to("cuda") / target_features.norm(dim=1, keepdim=True).to("cuda")
                        text_features = text_features / text_features.norm(dim=1, keepdim=True).to("cuda")
                        logits_per_image = image_features.to("cuda") @ text_features.t().to("cuda")
                        id = torch.argmax(logits_per_image.mean(axis=0))
                        if best_logits < logits_per_image[0][id]:
                            best_id = id + i * 1000
                            best_logits = logits_per_image[0][id]
                        del text_features
                        del inputs

                self.initial_condition = f"{vocab[best_id].split('<',1)[0]}"

                self.user_prompt += self.initial_condition

            else:
                self.initial_condition = None

            self.prompt_template = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
                {"role": "assistant", "content": self.model_prompt},
            ]
            self.llm_prompt = self.llm_tokenizer.apply_chat_template(
                self.prompt_template, return_tensors="pt"
            )[:,:-1]

    def get_feature(self, ):
        if not self.prompt_distillation:
            with torch.no_grad():
                curr_images = [torch.tensor(self.clip_image_transform(i)['pixel_values']) for i in self.target_images]
                curr_images = torch.concatenate(curr_images)
                # all_target_features = self.clip.get_image_features(pixel_values=curr_images)
                # all_target_features = all_target_features / all_target_features.norm(p=2, dim=-1, keepdim=True)
                vision_outputs = self.clip.vision_model(pixel_values=curr_images)
                pooled_output = vision_outputs['pooler_output']
                f_img = self.clip.visual_projection(pooled_output)
            self.register_buffer("all_target_features", f_img)
        else:
            with torch.no_grad():
                inputs = self.clip_tokenizer(self.target_prompt, max_length=76, truncation=True, padding=True, return_tensors="pt")
                f_txt = self.clip.get_text_features(**inputs)
                f_txt = f_txt / f_txt.norm(p=2, dim=-1, keepdim=True)
            self.register_buffer("all_target_features", f_txt)


    def reset(self):
        self.candidate = [[]]
        self.candidate_score = [[]] # torch.zeros(self.batch_size, self.clip_beam_size, device=self.llm.device)
        self.done_cnt = 0  # torch.zeros(self.batch_size, self.clip_beam_size, device=self.llm.device)
    def eval(self):
        super().eval()
        if not self.gen_prompt_only:
            self.sd_pipeline.to("cuda")
    def train(self, mode: bool = True):
        super().train(mode=mode)
        if not self.gen_prompt_only:
            self.sd_pipeline.to("cuda")
        self.freeze()

    def freeze(self):
        if self.freeze_model:
            self.llm.eval()
            self.llm.requires_grad_(False)
            for param in self.llm.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def inference(
            self,
            pixel_values: torch.Tensor=None,
            num_images_per_prompt: int=10,
            num_inference_steps: int=50,
            guidance_scale: float=9.0,
            style_transfer: bool=False,
            object: str=None
    ) -> Dict:


        generate_output = self.generate_prompt(
            pixel_values,
        )

        results = []

        if style_transfer:
            generated_prompt = generate_output['text'][0]
            prompt = f"{object} in the style of {generated_prompt}"

            if self.gen_prompt_only:
                images = None
            else:
                images = self.sd_pipeline(prompt=prompt, num_images_per_prompt=num_images_per_prompt,
                              guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images

            if self.get_similarity:
                similarity = self.measure_similarity(images)
            else:
                similarity = None

            results.append({"image": images, "initial_condition": self.initial_condition, "prompt": prompt,
                "input_prompt": generate_output["input_prompt"], "similarity": similarity, "object": object, "seed": self.seed})
        else:
            generated_prompt = generate_output['text'][0]
            input_prompt = generate_output['input_prompt'][0]

            if self.gen_prompt_only:
                images = None
            else:
                images = self.sd_pipeline(prompt=generated_prompt, num_images_per_prompt=num_images_per_prompt,
                                          guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images

            if self.get_similarity:
                similarity = self.measure_similarity(images)
            else:
                similarity = None

            results.append({"image": images, "initial_condition": self.initial_condition, "prompt": [generated_prompt],
                "input_prompt": input_prompt, "similarity": similarity, "seed": self.seed})

        return results

    def is_done(self):
        # if self.done_cnt > self.clip_beam_size * self.batch_size:
        if self.done_cnt > self.clip_beam_size:
            return 1
        return 0

    def length_penalty(self, length, alpha=1.2, min_length=5):

        return ((min_length + length)/ (min_length + 1 )) ** alpha

    def eos_check(self, indices:torch.Tensor, scores:torch.Tensor):
        mask = torch.eq(
            indices[:,-1],
            self.llm_tokenizer.eos_token_id
        )
        mask += torch.eq(
            indices[:, -1],
            13
        )

        if mask.sum() > 0:
            if len(indices[-1]) > self.min_length:
                for i, (s, m, ids) in enumerate(zip(scores, mask.reshape(scores.shape), indices.reshape(scores.shape[0],scores.shape[1],-1))):
                    ss = s[m].tolist()
                    if len(self.candidate_score[i]) > 0:
                        self.candidate_score[i] += ss
                        self.candidate[i] += [id for id in ids[m]]
                    else:
                        self.candidate_score[i] = ss
                        self.candidate[i] = [id for id in ids[m]]

                self.done_cnt += mask.reshape(scores.shape).sum()

            scores[mask.reshape(scores.shape)] = -torch.inf
        return scores

    @torch.no_grad()
    def generate_prompt(self, image: torch.Tensor=None) -> Dict:
        input_ids = self.llm_prompt.to(self.llm.device)

        llm_input_len = input_ids.shape[-1]
        llm_vocab_size = self.llm.vocab_size
        cummulative_scores = torch.zeros(1, device=self.llm.device)

        for curr_len in range(1, self.max_length+1):
            # llm decoding step
            llm_outputs = self.llm(
                input_ids,
                use_cache=False,
                return_dict=True
            )

            llm_scores = nn.functional.log_softmax(llm_outputs.logits[:,-1,:], dim=-1)
            llm_scores += cummulative_scores.view(input_ids.shape[0],-1) #P(x_{0:i})

            # Batch x Sequence x Vocab Size
            # llm_topk = torch.topk(llm_scores.view(self.batch_size,-1), dim=-1, k=self.llm_beam_size)
            llm_topk = torch.topk(llm_scores.view(1,-1), dim=-1, k=self.llm_beam_size)
            llm_topk_indices = llm_topk.indices % llm_vocab_size
            cummulative_scores = llm_topk.values
            llm_topk_beam_id = llm_topk.indices // llm_vocab_size
            llm_topk_batch_id = (llm_topk_beam_id + self.beam_offset.unsqueeze(-1)).view(-1)


            # if input_ids.shape[0] == self.batch_size: # in the first generation step.
            if input_ids.shape[0] == 1: # in the first generation step.
                # input_ids = input_ids.unsqueeze(1).expand(-1,self.clip_beam_size,-1).reshape(self.batch_size*self.clip_beam_size, -1)
                input_ids = input_ids.unsqueeze(1).expand(-1,self.clip_beam_size,-1).reshape(self.clip_beam_size, -1)

            # llm_beam_ids = torch.cat(
            #     (input_ids[llm_topk_batch_id].view(self.batch_size, self.llm_beam_size, -1),llm_topk_indices.unsqueeze(-1)),
            #     dim=-1
            # ) # batch x clip_beam x llm_beam x sequence
            llm_beam_ids = torch.cat(
                (input_ids[llm_topk_batch_id].view(1, self.llm_beam_size, -1),llm_topk_indices.unsqueeze(-1)),
                dim=-1
            ) # batch x clip_beam x llm_beam x sequence
            llm_beam_ids = llm_beam_ids.view(-1, llm_beam_ids.shape[-1])

            llm_gen_text = self.llm_tokenizer.batch_decode(
                llm_beam_ids[:,llm_input_len:],
                skip_special_tokens=True
            )

            # clip decoding step
            clip_input_ids = self.clip_tokenizer(llm_gen_text,padding=True, max_length=77, truncation=True)
            if hasattr(self, "all_target_features"):

                clip_text_embed = self.clip.get_text_features(
                    input_ids=torch.tensor(clip_input_ids['input_ids']).to(self.llm.device),
                    attention_mask=torch.tensor(clip_input_ids['attention_mask']).to(self.llm.device),
                )
                clip_text_embed = clip_text_embed / clip_text_embed.norm(p=2, dim=-1, keepdim=True)
                clip_logits_scale = self.clip.logit_scale.exp()
                # if self.target_images is not None and len(self.target_images) > 1:
                #     target_features = self.generate_random_target_feature()
                # else:
                target_features = self.all_target_features
                clip_logits = torch.matmul(target_features, clip_text_embed.t()) * clip_logits_scale
                # P(x_{0:i}|I)
                clip_scores = clip_logits.reshape(target_features.shape[0], -1,
                                                  self.llm_beam_size).log_softmax(dim=-1)

            else:
                clip_output = self.clip(
                    input_ids=torch.tensor(clip_input_ids['input_ids']).to(self.llm.device),
                    pixel_values=image,
                    attention_mask=torch.tensor(clip_input_ids['attention_mask']).to(self.llm.device),
                )
                clip_logits = clip_output.logits_per_image
                clip_scores = clip_logits.reshape(image.shape[0], -1,
                                                  self.llm_beam_size).log_softmax(dim=-1)


            total_score = self.llm_alpha * cummulative_scores + self.clip_alpha * clip_scores.mean(0)
            if self.length_cutoff:
                clip_input_length = torch.tensor(clip_input_ids['attention_mask']).sum(-1)

            # length_cutoff = (clip_input_length < self.min_length + 2) & (clip_input_length > self.max_length + 2)
                length_cutoff = (clip_input_length != curr_len + 2).unsqueeze(0)
                total_score[length_cutoff] = -torch.tensor(float('inf'))

            total_score = self.eos_check(llm_beam_ids[:, llm_input_len:], total_score)
            beam_index = (
                    total_score.topk(self.clip_beam_size, dim=-1).indices
                    + self.beam_expand_factor * self.beam_offset.unsqueeze(-1)
            ).view(-1)
            # cummulative_scores = cummulative_scores.reshape(self.batch_size * self.llm_beam_size,-1)
            cummulative_scores = cummulative_scores.reshape(1 * self.llm_beam_size,-1)
            # cummulative_scores = cummulative_scores[beam_index].reshape(self.batch_size * self.clip_beam_size,-1)
            cummulative_scores = cummulative_scores[beam_index].reshape(1 * self.clip_beam_size,-1)
            # input_ids = llm_beam_ids[beam_index].reshape(self.batch_size * self.clip_beam_size,-1)
            input_ids = llm_beam_ids[beam_index].reshape(1 * self.clip_beam_size,-1)

            # output_ids = input_ids.reshape(self.batch_size * self.clip_beam_size, -1)[0:10, llm_input_len:]
            output_ids = input_ids.reshape(1 * self.clip_beam_size, -1)[0:10, llm_input_len:]
            output_text = self.llm_tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )

            pprint(output_text)
            if self.is_done():
                break

        output_score = total_score[:, 0]
        # output_ids = input_ids.reshape(self.batch_size, self.clip_beam_size, -1)[:, 0, llm_input_len:]
        output_ids = input_ids.reshape(1, self.clip_beam_size, -1)[:, 0, llm_input_len:]

        # put the best-last decoded text into the candidate and pick the best
        # for b_i in range(self.batch_size):
        for b_i in range(1):
            self.candidate_score[b_i].append(output_score[b_i].item())
            self.candidate[b_i].append(output_ids[b_i])

            self.candidate_score[b_i] = [
                s / self.length_penalty(self.candidate[b_i][s_i].shape[-1]) for s_i, s in
                enumerate(self.candidate_score[b_i])
            ]

            b_i_best = self.candidate_score[b_i].index(max(self.candidate_score[b_i]))
            self.candidate[b_i] = self.candidate[b_i][b_i_best]

        output_text = self.llm_tokenizer.batch_decode(
            self.candidate,
            skip_special_tokens=True
        )
        # output_text[0] = "A photo of " + output_text[0]
        pprint(output_text)

        input_prompt = self.llm_tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=False
        )

        return {"text":output_text, "input_prompt": input_prompt}

    def find_nearest_tokens(self, curr_embeds):
        with torch.no_grad():
            bsz, emb_dim = curr_embeds.shape

            # Using the sentence transformers semantic search which is
            # a dot product exact kNN search between a set of
            # query vectors and a corpus of vectors
            curr_embeds = curr_embeds.reshape((-1, emb_dim))
            curr_embeds = normalize_embeddings(curr_embeds)  # queries

            embedding_matrix = self.clip.text_model.embeddings.token_embedding.weight
            embedding_matrix = normalize_embeddings(embedding_matrix)

            hits = semantic_search(curr_embeds, embedding_matrix,
                                   query_chunk_size=curr_embeds.shape[0],
                                   top_k=1000,
                                   )

            nn_indices = torch.tensor(
                [[h["corpus_id"] for h in hit] for hit in hits],
                device=curr_embeds.device
            )
            nn_indices = nn_indices.reshape(-1)

        return nn_indices

    def measure_similarity(self, output):
        with (torch.no_grad()):
            ori_batch = [self.preprocess_ref(i).unsqueeze(0) for i in self.target_images]
            if not self.clip:
                ori_batch = torch.concatenate(ori_batch).to(self.llm.device)
            else:
                ori_batch = torch.concatenate(ori_batch).to("cuda")

            gen_batch = [self.preprocess_ref(i).unsqueeze(0) for i in output]
            if not self.clip:
                gen_batch = torch.concatenate(gen_batch).to(self.llm.device)
            else:
                gen_batch = torch.concatenate(gen_batch).to("cuda")

            ori_feat = self.model_ref.encode_image(ori_batch)
            gen_feat = self.model_ref.encode_image(gen_batch)

            ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
            gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)

            return (ori_feat @ gen_feat.t()).mean().item()

