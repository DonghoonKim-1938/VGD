import hashlib
import os

import numpy as np
import open_clip
import requests
import torch
import math
from typing import Dict

from PIL import Image
from torch import nn
from tqdm import tqdm
from safetensors.numpy import load_file, save_file


class Interrogator(nn.Module):
    def __init__(self, cfg: Dict) -> None:

        super().__init__()
        """Initialize required components."""
        self.freeze_model = cfg["model"]["freeze_model"]
        self.quiet = cfg["model"]["quiet"]
        self.cache_path = cfg["model"]["cache_path"]
        self.download_cache = cfg["model"]["download_cache"]
        self.chunk_size = cfg["model"]["chunk_size"]
        self.flavor_intermediate_count = cfg["model"]["flavor_intermediate_count"]

        self.clip_name = cfg["model"]["clip"]
        self.clip_pretrain = cfg["model"]["clip_pretrain"]
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_name,
            pretrained=self.clip_pretrain,
            precision="fp16",
            jit=False,
            cache_dir=self.cache_path
        )
        self.tokenize = open_clip.get_tokenizer(self.clip_name)


        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribbble',
                 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount',
                 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']

        trending_list = [site for site in sites]
        trending_list.extend(["trending on " + site for site in sites])
        trending_list.extend(["featured on " + site for site in sites])
        trending_list.extend([site + " contest winner" for site in sites])

        self.bank_path = cfg["model"]["bank_path"]

        raw_artists = self.load_list(self.bank_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        self.artists = LabelTable(artists, "artists", self)
        self.flavors = LabelTable(self.load_list(self.bank_path, 'flavors.txt'), "flavors", self)
        self.mediums = LabelTable(self.load_list(self.bank_path, 'mediums.txt'), "mediums", self)
        self.movements = LabelTable(self.load_list(self.bank_path, 'movements.txt'), "movements", self)
        self.trendings = LabelTable(trending_list, "trendings", self)
        self.negative = LabelTable(self.load_list(self.bank_path, 'negative.txt'), "negative", self)

    def interrogate(self, image, min_flavors: int=8, max_flavors: int=32, caption: str=None, option: str='default') -> None:
        image_features = self.image_to_features(image)

        merged = self._merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self)
        flaves = merged.rank(image_features, self.flavor_intermediate_count)
        best_prompt, best_sim = caption, self.similarity(image_features, caption)
        best_prompt = self.chain(image_features, flaves, best_prompt, best_sim, min_count=min_flavors,
                                 max_count=max_flavors, desc="Flavor chain")

        fast_prompt = self.interrogate_fast(image, max_flavors, caption=caption)
        classic_prompt = self.interrogate_classic(image, max_flavors, caption=caption)
        if option == 'default':
            candidates = [caption, classic_prompt, fast_prompt, best_prompt]
        elif option == 'fast':
            candidates = [caption, fast_prompt]
        elif option == 'classic':
            candidates = [caption, classic_prompt]
        elif option == 'best':
            candidates = [caption, best_prompt]
        else:
            candidates = [caption]
        return candidates[np.argmax(self.similarities(image_features, candidates))]

    def interrogate_classic(self, image, max_flavors: int = 3, caption:str = None) -> str:
        """Classic mode creates a prompt in a standard format first describing the image,
        then listing the artist, trending, movement, and flavor text modifiers."""
        image_features = self.image_to_features(image)

        medium = self.mediums.rank(image_features, 1)[0]
        artist = self.artists.rank(image_features, 1)[0]
        trending = self.trendings.rank(image_features, 1)[0]
        movement = self.movements.rank(image_features, 1)[0]
        flaves = ", ".join(self.flavors.rank(image_features, max_flavors))

        if caption.startswith(medium):
            prompt = f"{caption} {artist}, {trending}, {movement}, {flaves}"
        else:
            prompt = f"{caption}, {medium} {artist}, {trending}, {movement}, {flaves}"

        return self._truncate_to_fit(prompt)

    def interrogate_fast(self, image, max_flavors: int = 32, caption:str = None) -> str:
        """Fast mode simply adds the top ranked terms after a caption. It generally results in
        better similarity between generated prompt and image than classic mode, but the prompts
        are less readable."""
        image_features = self.image_to_features(image)
        merged = self._merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self)
        tops = merged.rank(image_features, max_flavors)
        return self._truncate_to_fit(caption + ", " + ", ".join(tops))

    def load_list(self, data_path: str, filename):
        """Load a list of strings from a file."""
        if filename is not None:
            data_path = os.path.join(data_path, filename)
        with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
            items = [line.strip() for line in f.readlines()]
        return items

    def image_to_features(self, image: Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _merge_tables(self, tables, ci):
        m = LabelTable([], None, ci)
        for table in tables:
            m.labels.extend(table.labels)
            m.embeds.extend(table.embeds)
        return m

    def _truncate_to_fit(self, text: str) -> str:
        parts = text.split(', ')
        new_text = parts[0]
        for part in parts[1:]:
            if self._prompt_at_max_len(new_text + part, self.tokenize):
                break
            new_text += ', ' + part
        return new_text

    def chain(
            self,
            image_features: torch.Tensor,
            phrases,
            best_prompt: str = "",
            best_sim: float = 0,
            min_count: int = 8,
            max_count: int = 32,
            desc="Chaining",
            reverse: bool = False
    ) -> str:

        phrases = set(phrases)
        if not best_prompt:
            best_prompt = self.rank_top(image_features, [f for f in phrases], reverse=reverse)
            best_sim = self.similarity(image_features, best_prompt)
            phrases.remove(best_prompt)
        curr_prompt, curr_sim = best_prompt, best_sim

        def check(addition: str, idx: int) -> bool:
            nonlocal best_prompt, best_sim, curr_prompt, curr_sim
            prompt = curr_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if reverse:
                sim = -sim

            if sim > best_sim:
                best_prompt, best_sim = prompt, sim
            if sim > curr_sim or idx < min_count:
                curr_prompt, curr_sim = prompt, sim
                return True
            return False

        for idx in tqdm(range(max_count), desc=desc, disable=self.quiet):
            best = self.rank_top(image_features, [f"{curr_prompt}, {f}" for f in phrases], reverse=reverse)
            flave = best[len(curr_prompt) + 2:]
            if not check(flave, idx):
                break
            if self._prompt_at_max_len(curr_prompt, self.tokenize):
                break
            phrases.remove(flave)

        return best_prompt

    def rank_top(self, image_features: torch.Tensor, text_array, reverse: bool = False) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
            if reverse:
                similarity = -similarity
        return text_array[similarity.argmax().item()]


    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text]).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()

    def similarities(self, image_features: torch.Tensor, text_array):
        text_tokens = self.tokenize([text for text in text_array]).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity.T[0].tolist()

    def _prompt_at_max_len(self, text: str, tokenize) -> bool:
        tokens = tokenize([text])
        return tokens[0][-1] != 0



class LabelTable():
    def __init__(self, labels, desc: str, ci):
        self.clip_model = ci.clip_model
        self.chunk_size = ci.chunk_size
        self.embeds = []
        self.labels = labels
        self.tokenize = ci.tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()
        sanitized_name = f"{ci.clip_name}_{ci.clip_pretrain}"
        self.cache_path = ci.cache_path
        self.quiet = ci.quiet
        self.download_cache = ci.download_cache
        self.cache_path = ci.cache_path
        self._load_cached(desc, hash, sanitized_name)

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels) / self.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.quiet):
                text_tokens = self.tokenize(chunk)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if desc and self.cache_path:
                os.makedirs(self.cache_path, exist_ok=True)
                cache_filepath = os.path.join(self.cache_path, f"{sanitized_name}_{desc}.safetensors")
                tensors = {
                    "embeds": np.stack(self.embeds),
                    "hash": np.array([ord(c) for c in hash], dtype=np.int8)
                }
                save_file(tensors, cache_filepath)

    def _load_cached(self, desc: str, hash: str, sanitized_name: str) -> bool:
        if self.cache_path is None or desc is None:
            return False

        cached_safetensors = os.path.join(self.cache_path, f"{sanitized_name}_{desc}.safetensors")

        if self.download_cache and not os.path.exists(cached_safetensors):
            download_url = ('https://huggingface.co/pharmapsychotic/ci-preprocess/resolve/main/'+
                            f"{sanitized_name}_{desc}.safetensors")
            try:
                os.makedirs(self.cache_path, exist_ok=True)
                self.download_file(download_url, cached_safetensors, quiet=self.quiet)
            except Exception as e:
                print(f"Failed to download {download_url}")
                print(e)
                return False

        if os.path.exists(cached_safetensors):
            try:
                tensors = load_file(cached_safetensors)
            except Exception as e:
                print(f"Failed to load {cached_safetensors}")
                print(e)
                return False
            if 'hash' in tensors and 'embeds' in tensors:
                if np.array_equal(tensors['hash'], np.array([ord(c) for c in hash], dtype=np.int8)):
                    self.embeds = tensors['embeds']
                    if len(self.embeds.shape) == 2:
                        self.embeds = [self.embeds[i] for i in range(self.embeds.shape[0])]
                    return True

        return False

    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int = 1,
              reverse: bool = False) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
            if reverse:
                similarity = -similarity
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int = 1, reverse: bool = False):
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count, reverse=reverse)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels) / self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.quiet):
            start = chunk_idx * self.chunk_size
            stop = min(start + self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk, reverse=reverse)
            top_labels.extend([self.labels[start + i] for i in tops])
            top_embeds.extend([self.embeds[start + i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]

    def download_file(self, url: str, filepath: str, chunk_size: int = 4 * 1024 * 1024, quiet: bool = False):
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            return

        file_size = int(r.headers.get("Content-Length", 0))
        filename = url.split("/")[-1]
        progress = tqdm(total=file_size, unit="B", unit_scale=True, desc=filename, disable=quiet)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()