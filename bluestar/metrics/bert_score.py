import json
import os
import csv
from evaluate import load

def bert_score(prediction_path, save_path="./", save_file='bert_score.csv', dataset='mscoco'):
    bertscore = load("bertscore")
    total_score = []
    if dataset == 'mscoco':
        prompt_list_path = '/data/dataset/mscoco_100/prompt_list.json'
    elif dataset == 'lexica.art':
        prompt_list_path = '/data/dataset/lexica.art/prompt_list.json'
    else:
        prediction_path = dataset
    file_exists = os.path.isfile(os.path.join(save_path, save_file))
    with open(prediction_path) as f:
        reader = csv.DictReader(f)
        vgpg_results = list(reader)
    for result in vgpg_results:
        result_dict = {}
        prediction = [result['prompt'][2:-2]] if result['prompt'].startswith('"') or result['prompt'].startswith("'") or result['prompt'].startswith('[') \
            else [result['prompt']]
        with open (prompt_list_path, 'r') as f:
            prompt_list = json.load(f)
        reference_image = result['target_image'].split('/')[-1]
        reference = [prompt_list[reference_image]]
        score = bertscore.compute(predictions=prediction, references=reference, lang="en")
        file_exists = os.path.isfile(os.path.join(save_path, save_file))
        with open(os.path.join(os.path.join(save_path, save_file)), "a" if file_exists else "w", newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['reference_image', 'prediction', 'reference', 'precision', 'recall', 'f1'])
            writer.writerow([reference_image, prediction, reference, score['precision'][0], score['recall'][0], score['f1'][0]])

        result_dict['reference_image'] = reference_image
        result_dict['prediction'] = prediction
        result_dict['reference'] = reference
        result_dict['precision'] = score['precision'][0]
        result_dict['recall'] = score['recall'][0]
        result_dict['f1'] = score['f1'][0]
        total_score.append([result_dict])

    return total_score

def bert_score_dict(prediction_path, prompt_list_path='/data/dataset/mscoco_100/prompt_list.json',
               save_path="./", save_file='bert_score.csv'):
    bertscore = load("bertscore")
    total_score = []
    file_exists = os.path.isfile(os.path.join(save_path, save_file))

    with open(prediction_path, 'r') as f:
        prediction_results = json.load(f)
    for reference_image_path in prediction_results:
        result_dict = {}
        prediction = [prediction_results[reference_image_path]]
        with open (prompt_list_path, 'r') as f:
            prompt_list = json.load(f)
        reference_image = reference_image_path.split('/')[-1]
        reference = [prompt_list[reference_image]]

        score = bertscore.compute(predictions=prediction, references=reference, lang="en")

        file_exists = os.path.isfile(os.path.join(save_path, save_file))
        with open(os.path.join(os.path.join(save_path, save_file)), "a" if file_exists else "w", newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['reference_image', 'prediction', 'reference', 'precision', 'recall', 'f1'])
            writer.writerow([reference_image, prediction, reference, score['precision'][0], score['recall'][0], score['f1'][0]])

        result_dict['reference_image'] = reference_image
        result_dict['prediction'] = prediction
        result_dict['reference'] = reference
        result_dict['precision'] = score['precision'][0]
        result_dict['recall'] = score['recall'][0]
        result_dict['f1'] = score['f1'][0]
        total_score.append([result_dict])

    return total_score

