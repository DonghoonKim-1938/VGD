import os
import json
from collections import Counter

def image_classification_data_process(path,n_samples, dataset):
    filelist = [file for file in os.listdir(path) if file.endswith('.png')]
    classes = json.load(open(path+'/classes.json'))

    val = {}

    for file in filelist:
        filename = file.replace('.png', '')
        *_, prompt = filename.split('_', 1)
        class_name, *_ = prompt.split(' a type of ', 1)
        class_name = class_name.replace('a photo of ', '').replace(',','')
        label = '-1'

        if dataset == 'fgvc_aircraft':
            for i in range(len(classes)):
                class_full_name = classes[i]['manufacturer'].lower() + ' '+classes[i]['variant'].lower().replace('/','-')
                if class_name.lower() == class_full_name:
                    label = classes[i]['label']
                if class_name == 'Embraer Legacy 600':
                    label = 42
        else:
            for i in range(len(classes)):
                if class_name.lower() == classes[i]['class'].lower():
                    label = classes[i]['label']

        if label == '-1':
            raise Exception('wrong class: ', prompt, class_name)

        val[file] = label

    check, diff = image_classification_data_process_check(classes,val,n_samples)

    if check != 0:
        raise Exception('wrong n per class : ', diff)

    with open(path+'/val.json', 'w') as f:
        json.dump(val,f)


def image_classification_data_process_check(classes,val,n_samples):
    cnt = Counter(c['label'] for c in classes)
    cnt2 = Counter(val.values())

    cnt = {k: v * n_samples for k, v in cnt.items()}

    if cnt != cnt2:
        diff = {k: (cnt.get(k, 0), cnt2.get(k, 0)) for k in cnt.keys() | cnt2.keys() if cnt.get(k, 0) != cnt2.get(k, 0)}
        return -1, diff
    else:
        return 0, {}


def image_classification_data_process_inferece (path):
    folderlist = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]
    val = {}
    for folder in folderlist:
        label, class_name = folder.split('_',1)
        filelist = [file for file in os.listdir(os.path.join(path,folder)) if file.endswith('.png')]
        for file in filelist:
            file_path = os.path.join(folder,file)
            if 'cub' in path or 'CUB' in path:
                val[file_path] = int(label)-1
            else:
                val[file_path] = int(label)

    with open('/data/mjbae/txt2img_output'+'/val.json', 'w') as f:
        json.dump(val,f)
