import torch_fidelity
import json
import os
def kid_score(data_dir1, data_dir2, save_dir="./results.json", resize_size=128, kid_subset_size=100):

    metrics_dict = torch_fidelity.calculate_metrics(
            input1=data_dir1,
            input2=data_dir2,
            kid=True,
            samples_resize_and_crop=resize_size,
            kid_subset_size=kid_subset_size,
            samples_find_deep=True
        )
    metrics_dict['data_dir1'] = data_dir1
    metrics_dict['data_dir2'] = data_dir2


    with open(os.path.join(data_dir1,save_dir), 'w') as f:
        json.dump(metrics_dict, f)

    return metrics_dict