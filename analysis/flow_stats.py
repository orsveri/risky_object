import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def get_flow_feats(npz_file, filter_empty_frames=False):
    data = np.load(npz_file)
    flow = data['flow_feat']  # T, 31, 2048
    detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
    obj_labels = detection[:, :, 5]
    frame_labels = np.sum(obj_labels, axis=-1)
    frame_flow = flow[:, 0, :]
    obj_flow = flow[:, 1:, :]
    return frame_flow, obj_flow, frame_labels, obj_labels


def feat_stats(feat_collection):
    means = np.mean(feat_collection, axis=0)
    std = np.std(feat_collection, axis=0)
    return means, std


def dataset_stats(dir, file_list):
    frame_flow_feats = []
    obj_flow_feats = []
    frame_labels_feats = []
    obj_labels_feats = []
    for file in tqdm(file_list):
        frame_flow, obj_flow, frame_labels, obj_labels = get_flow_feats(os.path.join(dir, file))
        frame_flow_feats.append(frame_flow)
        obj_flow_feats.append(obj_flow)
        frame_labels_feats.append(frame_labels)
        obj_labels_feats.append(obj_labels)
    frame_flow_feats = np.array(frame_flow_feats)
    obj_flow_feats = np.array(obj_flow_feats)
    frame_labels_feats = np.array(frame_labels_feats).astype(bool)
    obj_labels_feats = np.array(obj_labels_feats).astype(bool)
    # accident frames
    acc_frame_nb = np.sum(frame_labels_feats)
    acc_feats = frame_flow_feats[frame_labels_feats]
    frame_acc_mean, frame_acc_std = feat_stats(acc_feats)
    # noacc frames
    noacc_frame_nb = frame_labels_feats.size - np.sum(frame_labels_feats)
    noacc_feats = frame_flow_feats[~frame_labels_feats]
    frame_noacc_mean, frame_noacc_std = feat_stats(noacc_feats)
    # acc objects
    acc_obj_nb = np.sum(obj_labels_feats)
    acc_feats = obj_flow_feats[obj_labels_feats]
    obj_acc_mean, obj_acc_std = feat_stats(acc_feats)
    # noacc objects
    noacc_obj_nb = obj_labels_feats.size - np.sum(obj_labels_feats)
    noacc_feats = obj_flow_feats[obj_labels_feats]
    obj_noacc_mean, obj_noacc_std = feat_stats(noacc_feats)
    return ((frame_acc_mean, frame_acc_std), (frame_noacc_mean, frame_noacc_std),
            (obj_acc_mean, obj_acc_std), (obj_noacc_mean, obj_noacc_std))



dir = "/mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature/val"

file_list = [f for f in os.listdir(dir) if f.endswith('.npz')]

# Calculate for datasets and then calculate RSD - Relative standard deviation + cosine distance
(frame_acc_stats, frame_noacc_stats, obj_acc_stats, obj_noacc_stats) = dataset_stats(dir=dir, file_list=file_list)



