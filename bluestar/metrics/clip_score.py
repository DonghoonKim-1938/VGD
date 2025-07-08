import glob
import os
import csv

import open_clip
import torch
from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def clip_score(target_images_path, output_path, save_path="./", save_file='clip_score.csv', clip_ref="ViT-g-14", clip_ref_pretrain="laion2b_s12b_b42k"):
    model_ref, _, preprocess_ref = open_clip.create_model_and_transforms(clip_ref, pretrained=clip_ref_pretrain)
    model_ref = model_ref.to("cuda")
    model_ref.eval()

    target_images = []
    total_score = []
    for img_extension in IMG_EXTENSIONS:
        target_images.extend(glob.glob(os.path.join(target_images_path, ('*' + img_extension))))
    target_images = [x for x in target_images if x.split('/')[-1] in os.listdir(output_path)]

    for target_image in target_images:
        output = []
        output_images = os.path.join(output_path, target_image.split('/')[-1])
        for img_extension in IMG_EXTENSIONS:
            output.extend(glob.glob(os.path.join(output_images, ('*' + img_extension))))
        output_image_list = [Image.open(i) for i in output]

        ori_batch = [preprocess_ref(Image.open(target_image)).unsqueeze(0)]
        ori_batch = torch.concatenate(ori_batch).to("cuda")

        gen_batch = [preprocess_ref(i).unsqueeze(0) for i in output_image_list]
        gen_batch = torch.concatenate(gen_batch).to("cuda")

        ori_feat = model_ref.encode_image(ori_batch)
        gen_feat = model_ref.encode_image(gen_batch)

        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)

        clip_score = (ori_feat @ gen_feat.t()).mean().item()

        file_exists = os.path.isfile(os.path.join(save_path, save_file))
        result_dict = {}
        with open(os.path.join(os.path.join(save_path, save_file)), "a" if file_exists else "w", newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['target_image', 'output', 'clip_score'])
            writer.writerow([target_image, output_images, clip_score])

        result_dict['target_image'] = target_image
        result_dict['output'] = output_images
        result_dict['clip_score'] = clip_score

        total_score.append([result_dict])

        del ori_feat
        del ori_batch
        del gen_batch
        del gen_feat

    return total_score


