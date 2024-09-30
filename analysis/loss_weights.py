import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def get_obj_labels(npz_file, filter_empty_frames=False):
    data = np.load(npz_file)
    detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
    actual_detection = detection[detection[:, :, 0] > 0]
    obj_labels = actual_detection[:, 5]
    return obj_labels


def dataset_stats(dir, file_list):
    acc_count = 0
    noacc_count = 0
    for file in tqdm(file_list):
        obj_labels = get_obj_labels(os.path.join(dir, file))
        nb_acc = np.sum(obj_labels)
        nb_noacc = obj_labels.size - nb_acc
        acc_count += nb_acc
        noacc_count += nb_noacc
    sum = acc_count + noacc_count
    acc_weights = sum / acc_count
    noacc_weights = sum / noacc_count
    wmax = max(acc_weights, noacc_weights)
    print(f"DATA {dir}\n\tAcc: {int(acc_count)}, Noacc: {int(noacc_count)}.")
    print(f"\tnoacc weights: {noacc_weights/wmax}, acc weights: {acc_weights/wmax}\n")



dir = "/mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature/train"
dataset_stats(
    dir=dir,
    file_list=[f for f in os.listdir(dir) if f.endswith('.npz')]
)

dir = "/mnt/experiments/sorlova/datasets/ROL/AMNet_DoTA/train"
dataset_stats(
    dir=dir,
    file_list=[f for f in os.listdir(dir) if f.endswith('.npz')]
)

dir = "/mnt/experiments/sorlova/datasets/GTACrash/AMNet_feats/train"
dataset_stats(
    dir=dir,
    file_list=[f for f in os.listdir(dir) if f.endswith('.npz')]
)


