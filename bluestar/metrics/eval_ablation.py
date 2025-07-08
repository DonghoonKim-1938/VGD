#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel


import json
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torchvision.transforms.functional as TF
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
import re

# prompt_token = r"(?<![0-9])-5(?![0-9])"
prompt_token = r"-4"
subject_names_cub = ['071_Long_tailed_Jaeger', '057_Rose_breasted_Grosbeak', '106_Horned_Puffin', '006_Least_Auklet', '037_Acadian_Flycatcher', '169_Magnolia_Warbler', '067_Anna_Hummingbird', '180_Wilson_Warbler', '158_Bay_breasted_Warbler', '197_Marsh_Wren', '041_Scissor_tailed_Flycatcher', '181_Worm_eating_Warbler', '086_Pacific_Loon', '107_Common_Raven', '090_Red_breasted_Merganser', '146_Forsters_Tern', '020_Yellow_breasted_Chat', '045_Northern_Fulmar', '033_Yellow_billed_Cuckoo', '153_Philadelphia_Vireo', '109_American_Redstart', '066_Western_Gull', '138_Tree_Swallow', '010_Red_winged_Blackbird', '055_Evening_Grosbeak', '114_Black_throated_Sparrow', '076_Dark_eyed_Junco', '161_Blue_winged_Warbler', '001_Black_footed_Albatross', '112_Great_Grey_Shrike', '077_Tropical_Kingbird', '131_Vesper_Sparrow', '104_American_Pipit', '003_Sooty_Albatross', '139_Scarlet_Tanager', '063_Ivory_Gull', '130_Tree_Sparrow', '177_Prothonotary_Warbler', '069_Rufous_Hummingbird', '195_Carolina_Wren', '165_Chestnut_sided_Warbler', '016_Painted_Bunting', '088_Western_Meadowlark', '061_Heermann_Gull', '023_Brandt_Cormorant', '196_House_Wren', '129_Song_Sparrow', '031_Black_billed_Cuckoo', '123_Henslow_Sparrow', '173_Orange_crowned_Warbler', '046_Gadwall', '147_Least_Tern', '145_Elegant_Tern', '098_Scott_Oriole', '185_Bohemian_Waxwing', '175_Pine_Warbler', '178_Swainson_Warbler', '075_Green_Jay', '038_Great_Crested_Flycatcher', '072_Pomarine_Jaeger', '160_Black_throated_Blue_Warbler', '164_Cerulean_Warbler', '189_Red_bellied_Woodpecker', '192_Downy_Woodpecker', '156_White_eyed_Vireo', '043_Yellow_bellied_Flycatcher', '017_Cardinal', '119_Field_Sparrow', '162_Canada_Warbler', '039_Least_Flycatcher', '042_Vermilion_Flycatcher', '026_Bronzed_Cowbird', '121_Grasshopper_Sparrow', '108_White_necked_Raven', '082_Ringed_Kingfisher', '040_Olive_sided_Flycatcher', '084_Red_legged_Kittiwake', '035_Purple_Finch', '007_Parakeet_Auklet', '032_Mangrove_Cuckoo', '083_White_breasted_Kingfisher', '148_Green_tailed_Towhee', '171_Myrtle_Warbler', '118_House_Sparrow', '094_White_breasted_Nuthatch', '183_Northern_Waterthrush', '135_Bank_Swallow', '110_Geococcyx', '166_Golden_winged_Warbler', '137_Cliff_Swallow', '122_Harris_Sparrow', '134_Cape_Glossy_Starling', '096_Hooded_Oriole', '199_Winter_Wren', '101_White_Pelican', '099_Ovenbird', '141_Artic_Tern', '022_Chuck_will_Widow', '064_Ring_billed_Gull', '087_Mallard', '051_Horned_Grebe', '052_Pied_billed_Grebe', '125_Lincoln_Sparrow', '132_White_crowned_Sparrow', '058_Pigeon_Guillemot', '198_Rock_Wren', '085_Horned_Lark', '005_Crested_Auklet', '009_Brewer_Blackbird', '070_Green_Violetear', '200_Common_Yellowthroat', '065_Slaty_backed_Gull', '029_American_Crow', '187_American_Three_toed_Woodpecker', '163_Cape_May_Warbler', '140_Summer_Tanager', '059_California_Gull', '182_Yellow_Warbler', '004_Groove_billed_Ani', '054_Blue_Grosbeak', '079_Belted_Kingfisher', '013_Bobolink', '188_Pileated_Woodpecker', '044_Frigatebird', '124_Le_Conte_Sparrow', '093_Clark_Nutcracker', '176_Prairie_Warbler', '068_Ruby_throated_Hummingbird', '024_Red_faced_Cormorant', '048_European_Goldfinch', '036_Northern_Flicker', '151_Black_capped_Vireo', '074_Florida_Jay', '011_Rusty_Blackbird', '184_Louisiana_Waterthrush', '190_Red_cockaded_Woodpecker', '115_Brewer_Sparrow', '053_Western_Grebe', '143_Caspian_Tern', '170_Mourning_Warbler', '152_Blue_headed_Vireo', '018_Spotted_Catbird', '194_Cactus_Wren', '002_Laysan_Albatross', '159_Black_and_white_Warbler', '019_Gray_Catbird', '015_Lazuli_Bunting', '027_Shiny_Cowbird', '097_Orchard_Oriole', '174_Palm_Warbler', '034_Gray_crowned_Rosy_Finch', '014_Indigo_Bunting', '030_Fish_Crow', '179_Tennessee_Warbler', '167_Hooded_Warbler', '150_Sage_Thrasher', '100_Brown_Pelican', '157_Yellow_throated_Vireo', '103_Sayornis', '049_Boat_tailed_Grackle', '008_Rhinoceros_Auklet', '021_Eastern_Towhee', '078_Gray_Kingbird', '062_Herring_Gull', '080_Green_Kingfisher', '149_Brown_Thrasher', '095_Baltimore_Oriole', '144_Common_Tern', '133_White_throated_Sparrow', '089_Hooded_Merganser', '025_Pelagic_Cormorant', '073_Blue_Jay', '012_Yellow_headed_Blackbird', '154_Red_eyed_Vireo', '172_Nashville_Warbler', '102_Western_Wood_Pewee', '193_Bewick_Wren', '127_Savannah_Sparrow', '050_Eared_Grebe', '105_Whip_poor_Will', '186_Cedar_Waxwing', '081_Pied_Kingfisher', '117_Clay_colored_Sparrow', '028_Brown_Creeper', '111_Loggerhead_Shrike', '091_Mockingbird', '191_Red_headed_Woodpecker', '116_Chipping_Sparrow', '155_Warbling_Vireo', '136_Barn_Swallow', '126_Nelson_Sharp_tailed_Sparrow', '168_Kentucky_Warbler', '092_Nighthawk', '142_Black_Tern', '128_Seaside_Sparrow', '113_Baird_Sparrow', '120_Fox_Sparrow', '060_Glaucous_winged_Gull', '047_American_Goldfinch', '056_Pine_Grosbeak']
subject_names_stanford_dogs = ['51_soft-coated_wheaten_terrier', '97_Eskimo_dog', '25_Saluki', '58_Chesapeake_Bay_retriever', '20_Italian_greyhound', '116_Mexican_hairless', '35_Norwich_terrier', '80_collie', '114_miniature_poodle', '72_schipperke', '90_EntleBucher', '108_chow', '82_Bouvier_des_Flandres', '26_Scottish_deerhound', '87_Greater_Swiss_Mountain_dog', '45_miniature_schnauzer', '31_Border_terrier', '13_bluetick', '43_Dandie_Dinmont', '60_vizsla', '61_English_setter', '113_toy_poodle', '8_Rhodesian_ridgeback', '83_Rottweiler', '34_Norfolk_terrier', '42_Australian_terrier', '107_Pomeranian', '92_bull_mastiff', '15_Walker_hound', '74_malinois', '38_Lakeland_terrier', '19_Irish_wolfhound', '6_papillon', '109_keeshond', '47_standard_schnauzer', '50_silky_terrier', '115_standard_poodle', '105_Great_Pyrenees', '10_basset', '14_black-and-tan_coonhound', '102_pug', '84_German_shepherd', '59_German_short-haired_pointer', '119_African_hunting_dog', '21_whippet', '96_Saint_Bernard', '63_Gordon_setter', '46_giant_schnauzer', '18_borzoi', '55_curly-coated_retriever', '99_Siberian_husky', '98_malamute', '71_kuvasz', '95_Great_Dane', '28_Staffordshire_bullterrier', '48_Scotch_terrier', '100_affenpinscher', '37_wire-haired_fox_terrier', '118_dhole', '66_English_springer', '101_basenji', '89_Appenzeller', '78_Old_English_sheepdog', '7_toy_terrier', '65_clumber', '75_briard', '11_beagle', '40_Airedale', '76_kelpie', '0_Chihuahua', '53_Lhasa', '62_Irish_setter', '1_Japanese_spaniel', '23_Norwegian_elkhound', '69_Sussex_spaniel', '33_Irish_terrier', '32_Kerry_blue_terrier', '29_American_Staffordshire_terrier', '54_flat-coated_retriever', '110_Brabancon_griffon', '70_Irish_water_spaniel', '85_Doberman', '91_boxer', '5_Blenheim_spaniel', '4_Shih-Tzu', '112_Cardigan', '111_Pembroke', '12_bloodhound', '88_Bernese_mountain_dog', '39_Sealyham_terrier', '106_Samoyed', '81_Border_collie', '103_Leonberg', '79_Shetland_sheepdog', '36_Yorkshire_terrier', '117_dingo', '56_golden_retriever', '22_Ibizan_hound', '17_redbone', '86_miniature_pinscher', '104_Newfoundland', '2_Maltese_dog', '9_Afghan_hound', '16_English_foxhound', '68_cocker_spaniel', '77_komondor', '52_West_Highland_white_terrier', '49_Tibetan_terrier', '94_French_bulldog', '57_Labrador_retriever', '93_Tibetan_mastiff', '27_Weimaraner', '30_Bedlington_terrier', '41_cairn', '64_Brittany_spaniel', '67_Welsh_springer_spaniel', '44_Boston_bull', '3_Pekinese', '73_groenendael', '24_otterhound']
subject_names_fgvc = ['61_PA-28', '15_An-12', '1_Boeing_727', '13_ATR-42', '36_Dash_8', '48_Fokker_100', '46_Falcon_2000', '32_DC-9', '60_King_Air', '37_DR-400', '25_Cessna_208', '53_Hawk_T1', '7_A300', '66_Tornado', '39_Embraer_E-Jet', '11_A340', '14_ATR-72', '57_MD-80', '12_A380', '40_EMB-120', '30_DC-6', '3_Boeing_747', '69_Yak-42', '27_Challenger_600', '23_CRJ-700', '33_DH-82', '67_Tu-134', '8_A310', '52_Gulfstream', '19_Boeing_717', '44_F-16', '22_CRJ-200', '24_Cessna_172', '41_Embraer_ERJ_145', '21_C-47', '62_SR-20', '35_DHC-6', '38_Dornier_328', '9_A320', '18_Beechcraft_1900', '10_A330', '47_Falcon_900', '4_Boeing_757', '20_C-130', '64_Saab_340', '50_Fokker_70', '2_Boeing_737', '29_DC-3', '17_BAE-125', '45_F-A-18', '43_Eurofighter_Typhoon', '63_Saab_2000', '34_DHC-1', '54_Il-76', '59_Metroliner', '42_Embraer_Legacy_600', '49_Fokker_50', '6_Boeing_777', '28_DC-10', '56_MD-11', '5_Boeing_767', '51_Global_Express', '26_Cessna_Citation', '68_Tu-154', '58_MD-90', '55_L-1011', '31_DC-8', '0_Boeing_707', '65_Spitfire', '16_BAE_146']
subject_names_imagenet = ['n03871628', 'n02640242', 'n02115913', 'n02111129', 'n12985857', 'n02641379', 'n03599486', 'n04493381', 'n02112706', 'n04037443', 'n02486261', 'n03825788', 'n04067472', 'n01608432', 'n02107908', 'n01688243', 'n02110185', 'n03803284', 'n02894605', 'n03895866', 'n02892201', 'n01755581', 'n03954731', 'n04428191', 'n03141823', 'n02276258', 'n03967562', 'n02110063', 'n01631663', 'n02120505', 'n04515003', 'n01873310', 'n03794056', 'n03017168', 'n04264628', 'n03991062', 'n03742115', 'n01739381', 'n04532106', 'n02814533', 'n02094258', 'n01632458', 'n04367480', 'n03887697', 'n04311174', 'n02229544', 'n03642806', 'n03773504', 'n02814860', 'n01797886', 'n03207743', 'n02442845', 'n01729322', 'n03868863', 'n03218198', 'n02097298', 'n12620546', 'n02690373', 'n02113186', 'n04456115', 'n03584829', 'n03337140', 'n01847000', 'n03447721', 'n02443114', 'n02113023', 'n02840245', 'n04371774', 'n02704792', 'n04277352', 'n03666591', 'n04357314', 'n04579145', 'n02113712', 'n03983396', 'n02747177', 'n04392985', 'n01644900', 'n02879718', 'n04372370', 'n10148035', 'n02536864', 'n02791270', 'n03710721', 'n02797295', 'n01990800', 'n01776313', 'n01930112', 'n02108000', 'n02490219', 'n03016953', 'n04296562', 'n13040303', 'n01669191', 'n02087046', 'n02869837', 'n02123159', 'n02927161', 'n04152593', 'n07579787', 'n13133613', 'n01734418', 'n04604644', 'n04118776', 'n02101556', 'n02492035', 'n02930766', 'n03483316', 'n03841143', 'n02497673', 'n02093256', 'n04560804', 'n02325366', 'n03929660', 'n03804744', 'n01735189', 'n02097658', 'n02979186', 'n03884397', 'n01756291', 'n04136333', 'n02089867', 'n02096437', 'n01728572', 'n03032252', 'n01748264', 'n01955084', 'n03126707', 'n01682714', 'n02256656', 'n03947888', 'n04125021', 'n02321529', 'n12768682', 'n01980166', 'n02115641', 'n02825657', 'n01685808', 'n01728920', 'n02096177', 'n01632777', 'n03134739', 'n03868242', 'n02817516', 'n03662601', 'n04141975', 'n03673027', 'n13037406', 'n12998815', 'n04033901']
subject_names_dreambooth = ['robot_toy', 'dog2', 'can', 'dog3', 'pink_sunglasses', 'duck_toy', 'cat', 'shiny_sneaker', 'backpack', 'rc_car', 'monster_toy', 'candle', 'dog', 'clock', 'red_cartoon', 'vase', 'fancy_boot', 'backpack_dog', 'grey_sloth_plushie', 'berry_bowl', 'dog5', 'bear_plushie', 'teapot', 'dog6', 'cat2', 'dog7', 'wolf_plushie', 'poop_emoji', 'dog8', 'colorful_sneaker']
class PromptDatasetCLIP(Dataset):
    def __init__(self, image_dir, json_file, tokenizer, processor, epoch=None):
        with open(json_file, 'r') as json_file:
            metadata_dict  = json.load(json_file)
        
        self.image_dir = image_dir
        self.image_lst = []
        self.prompt_lst = []
        for key, value in metadata_dict.items():
            if re.search(prompt_token, value['data_dir']):
                if epoch is not None:
                    data_dir = os.path.join(self.image_dir, value['data_dir'], str(epoch))
                else:
                    data_dir = os.path.join(self.image_dir, value['data_dir'])
                image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]
                self.image_lst.extend(image_files[:4])
                class_prompts = [value['instance_prompt']] * len(image_files)
                self.prompt_lst.extend(class_prompts[:4])
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs


class PairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, processor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        self.image_files_B = []
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if
                              f.endswith(".JPEG") or f.endswith(".jpg") or f.endswith(".png")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_b = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if
                              f.endswith(".JPEG") or f.endswith(".jpg") or f.endswith(".png")]
        self.image_files_B.extend(self.image_files_b[:10])
        #
        # self.image_files_B = []
        # # Get image files from each subfolder in data A
        # for subfolder in os.listdir(data_dir_B):
        #     if subject in subfolder:
        #         # if re.search(prompt_token, subfolder):
        #         #     if epoch is not None:
        #         #         data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
        #         #     else:
        #             data_dir_B = os.path.join(self.data_dir_B, subfolder)
        #             image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
        #             self.image_files_B.extend(image_files_b[:4])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A)*len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        # image_A_filename = self.image_files_A[index_A].split('/')[-1]
        # if 'imagenet' in self.data_dir_B:
        #     index_B = self.image_files_B.index(os.path.join(self.data_dir_B, image_A_filename.replace('.png','.JPEG')))
        # else:
        #     index_B = self.image_files_B.index(os.path.join(self.data_dir_B, image_A_filename.replace('.png','.jpg')))
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(val == 0 for val in extrema_A) or all(val == 0 for val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B

        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if
                              f.endswith(".JPEG") or f.endswith(".jpg") or f.endswith(".png")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if
                              f.endswith(".png") or f.endswith(".JPEG") or f.endswith(".jpg") or f.endswith(".png")]

        # self.data_dir_A = data_dir_A
        # self.data_dir_B = data_dir_B
        #
        # self.data_dir_A = os.path.join(self.data_dir_A, subject)
        # self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]
        #
        # # subject = subject + '-'
        # self.image_files_B = []
        #
	    # # Get image files from each subfolder in data A
        # for subfolder in os.listdir(data_dir_B):
        #     if subject in subfolder:
        #         if re.search(prompt_token, subfolder):
        #             if epoch is not None:
        #                 data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
        #             else:
        #                 data_dir_B = os.path.join(self.data_dir_B, subfolder)
        #             image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
        #             self.image_files_B.extend(image_files_b[:4])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(val == 0 for val in extrema_A) or all(val == 0 for val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class SelfPairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir, processor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.processor(images=image_A, return_tensors="pt")
        inputs_B = self.processor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir, feature_extractor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
        inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B




def clip_text(image_dir, epoch=None):
    criterion = 'clip_text'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the text features
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = PromptDatasetCLIP(image_dir, 'metadata.json', tokenizer, processor, epoch)
    dataloader = DataLoader(dataset, batch_size=32)

    similarity = []
    for i in tqdm(range(len(dataset))):
        image_inputs, prompt_inputs = dataset[i]
        if image_inputs is not None and prompt_inputs is not None:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
            # print(prompt_inputs)
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**prompt_inputs)

            sim = cosine_similarity(image_features, text_features)

            #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            #logit_scale = model.logit_scale.exp()
            #sim = torch.matmul(text_features, image_features.t()) * logit_scale
            similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def clip_image(data_dir, image_dir, epoch=None, dataset='cub'):
    criterion = 'clip_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    similarity = []

    if dataset == 'cub':
        subject_names = subject_names_cub
    elif dataset == 'stanford_dogs':
        subject_names = subject_names_stanford_dogs
    elif dataset == 'fgvc':
        subject_names = subject_names_fgvc
    elif dataset == 'imagenet':
        subject_names = subject_names_imagenet
    elif dataset == 'dreambooth':
        subject_names = subject_names_dreambooth
    else:
        raise ValueError(f'unknown dataset: {dataset}')

    for subject in subject_names:
        dataset = PairwiseImageDatasetCLIP(subject, data_dir, image_dir, processor, epoch)
        # dataset = SelfPairwiseImageDatasetCLIP(subject, './data', processor)

        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                image_A_features = model.get_image_features(**inputs_A)
                image_B_features = model.get_image_features(**inputs_B)

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
            
                logit_scale = model.logit_scale.exp()
                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def dino(data_dir, image_dir, epoch=None, dataset='cub'):
    criterion = 'dino'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')

    if dataset == 'cub':
        subject_names = subject_names_cub
    elif dataset == 'stanford_dogs':
        subject_names = subject_names_stanford_dogs
    elif dataset == 'fgvc':
        subject_names = subject_names_fgvc
    elif dataset == 'imagenet':
        subject_names = subject_names_imagenet
    else:
        raise ValueError(f'unknown dataset: {dataset}')

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetDINO(subject, data_dir, image_dir, feature_extractor, epoch)
        # dataset = SelfPairwiseImageDatasetDINO(subject, './data', feature_extractor)

        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                outputs_A = model(**inputs_A)
                image_A_features = outputs_A.last_hidden_state[:, 0, :]

                outputs_B = model(**inputs_B)
                image_B_features = outputs_B.last_hidden_state[:, 0, :]

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion
#
# if __name__ == "__main__":
#     image_dir = '/data/mjbae/txt2img_output/SD-IPC/full/CUB/'
#     data_dir = '/data/dataset/caltech-bird/split/test/'
#     epoch = None
#
#     # 3 different evaluation metrics
#     # sim, criterion = clip_text(image_dir, epoch)
#     # sim, criterion = dino(image_dir, epoch)
#     sim, criterion = clip_image(image_dir, data_dir, epoch)
#
#     if epoch:
#         name = image_dir + '-' + str(epoch)
#     else:
#         name = image_dir
#
#     filename = "results_evaluation.txt"  # the name of the file to save the value to
#     # Check if file already exists
#     file_exists = os.path.isfile(filename)
#
#     # Open the file in append mode if it exists, otherwise create a new file
#     with open(filename, "a" if file_exists else "w") as file:
#         # If the file exists, add a new line before writing the new data
#         if file_exists:
#             file.write("\n")
#         # Write the name and value as a comma-separated string to the file
#         file.write(f"{criterion},{name},{sim}")


def eval_ablation(image_dir, data_dir, epoch=None, metric='clip_image', dataset='cub'):
    if metric == 'clip_text':
        sim, criterion = clip_text(image_dir, epoch, dataset)
    elif metric == 'clip_image':
        sim, criterion = clip_image(image_dir, data_dir, epoch, dataset)
    elif metric == 'dino':
        sim, criterion = dino(data_dir, image_dir, epoch, dataset)
    else:
        raise ValueError(f'Unknown metric: {metric}')

    if epoch:
        name = image_dir + '-' + str(epoch)
    else:
        name = image_dir + ',' + data_dir

    filename = "results_evaluation.txt"  # the name of the file to save the value to
    # Check if file already exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode if it exists, otherwise create a new file
    with open(filename, "a" if file_exists else "w") as file:
        # If the file exists, add a new line before writing the new data
        if file_exists:
            file.write("\n")
        # Write the name and value as a comma-separated string to the file
        file.write(f"{criterion},{name},{sim}")


    return [criterion, name, sim]

#
