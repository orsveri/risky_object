import pandas as pd
import numpy as np
from natsort import natsorted
import glob
import os
import io
import sys
import zipfile

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.model import FeatureExtractor
from PIL import Image
import argparse
import logging
device = ("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.Resize(224),
        transforms.ToTensor(),
     ]
)
to_pil_image = transforms.ToPILImage()


def log_information(vid_id, video_frame, track_id, e):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('Error_log.log')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    logger.info(
        f"Error: {e}------Video_id-{vid_id}--video_frame: {video_frame} at tracking_id {track_id}")
    return


def get_args():
    parser = argparse.ArgumentParser()
    # path to a dataset folder, containing subfolders with clips
    parser.add_argument("--path")
    args = parser.parse_args()
    return args


def featureExt(extractor, image):
    # image = Image.open(img_path)
    image = transform(image)  # 3 x 224 x 224 (for the frame, for object it varies)
    # print('shape image : ', image.shape)
    image = torch.unsqueeze(image, 0).float().to(device=device)
    # print(image.shape)
    with torch.no_grad():
        feat = extractor(image)  # 1 x 2048 x 1 x 1
        feat = torch.squeeze(feat, 2)  # 1 x 2048 x 1
        feat = torch.squeeze(feat, 2)  # 1 x 2048
    return feat  # 1x 2048


def obj_featureExt(extractor, image, xmin, ymin, xmax, ymax):
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    cropped_image = transform(cropped_image)
    # print('shape image : ', image.shape)
    cropped_image = torch.unsqueeze(cropped_image, 0).float().to(device=device)
    # print(image.shape)
    with torch.no_grad():
        feat = extractor(cropped_image)  # 1 x 2048 x 1 x 1
        feat = torch.squeeze(feat, 2)  # 1 x 2048 x 1
        feat = torch.squeeze(feat, 2)  # 1 x 2048
    return feat  # 1x 2048


def bbox_to_imroi(video_frame, bbox):
    image = Image.open(video_frame)
    image = transform(image)
    imroi = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]  # (x1:x2, y1:y2)
    return imroi


def main():

    args = get_args()
    dataset_path = args.path

    clip_folders = natsorted(os.listdir(dataset_path))
    clip_folders = [os.path.join(dataset_path, cf) for cf in clip_folders if os.path.isdir(os.path.join(dataset_path, cf))]

    scaling_w_rgb = 3.17  # 710/224
    scaling_w_flow = 3.18
    scaling_h_rgb = scaling_h_flow = 1.79  # 400/224
    count = 0

    extractor = FeatureExtractor().to(device=device)
    extractor.eval()

    for clip_folder in tqdm(clip_folders):
        # All files
        dets = pd.read_csv(os.path.join(clip_folder, "detections.csv"))
        rgb_zip = os.path.join(clip_folder, "rgb.zip")
        flow_zip = os.path.join(clip_folder, "flow.zip")
        out_file = os.path.join(clip_folder, f"{os.path.basename(clip_folder)}.npz")
        frames = natsorted(dets["frame"].unique())
        N = len(frames) - 1
        # N frames, 30 maximum objects, 6: 1-> track_id, (2,3,4,5)-> (y1,x1;y2,x2), 6-> object serial number in each frame
        detections = np.zeros((N, 30, 6), dtype=np.float32)
        # 100 frames, 1 global frame-level feature + 30 object level feature , resnet50 feat dimension 2048
        feature_rgb = np.zeros((N, 31, 2048), dtype=np.float32)
        feature_flow = np.zeros((N, 31, 2048), dtype=np.float32)

        # Fill in detections
        frames = natsorted(dets["frame"].unique())
        first = frames.pop(0)  # We do not use the first frame because of the optical flow

        for i, frame in enumerate(frames):
            frame_dets = dets[dets["frame"] == frame]
            frame_dets = frame_dets[["track_id", "xmin", "ymin", "xmax", "ymax", "label"]].to_numpy()
            frame_dets = np.pad(frame_dets, ((0, 30-frame_dets.shape[0]), (0, 0)), mode='constant', constant_values=0)
            detections[i] = frame_dets
        # RGB frames
        with zipfile.ZipFile(rgb_zip, 'r') as z:
            for j, frame in enumerate(frames):
                image = None
                image_file = f"{str(frame).zfill(6)}.jpg"
                with z.open(image_file) as f:
                    image = Image.open(io.BytesIO(f.read()))
                w_, h_ = image.size
                # First, the global frame feature
                feat = featureExt(extractor=extractor, image=image)
                # feat = feat.detach().numpy()
                feat = feat.cpu().numpy() if feat.is_cuda else feat.detach().numpy()
                feature_rgb[j, 0, :] = feat
                # Then, the object features
                for iobj, det in enumerate(detections[j]):
                    coord = det[1:5].copy()
                    coord[0] = round(coord[0] * w_)
                    coord[1] = round(coord[1] * h_)
                    coord[2] = round(coord[2] * w_)
                    coord[3] = round(coord[3] * h_)
                    if (coord[2] - coord[0]) * (coord[3] - coord[1]) == 0:
                        continue
                    xmin, ymin, xmax, ymax = coord
                    feat = obj_featureExt(extractor=extractor, image=image, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                    feat = feat.cpu().numpy() if feat.is_cuda else feat.detach().numpy()
                    feature_rgb[j, iobj+1, :] = feat

        # Flow frames
        with zipfile.ZipFile(flow_zip, 'r') as z:
            for k, frame in enumerate(frames):
                image = None
                image_file = f"{str(frame).zfill(6)}.png"
                with z.open(image_file) as f:
                    image = Image.open(io.BytesIO(f.read()))
                w_, h_ = image.size
                # First, the global frame feature
                feat = featureExt(extractor=extractor, image=image)
                # feat = feat.detach().numpy()
                feat = feat.cpu().numpy() if feat.is_cuda else feat.detach().numpy()
                feature_flow[k, 0, :] = feat
                # Then, the object features
                for iobj, det in enumerate(detections[k]):
                    coord = det[1:5].copy()
                    coord[0] = round(coord[0] * w_)
                    coord[1] = round(coord[1] * h_)
                    coord[2] = round(coord[2] * w_)
                    coord[3] = round(coord[3] * h_)
                    if (coord[2] - coord[0]) * (coord[3] - coord[1]) == 0:
                        continue
                    xmin, ymin, xmax, ymax = coord
                    feat = obj_featureExt(extractor=extractor, image=image, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                    feat = feat.cpu().numpy() if feat.is_cuda else feat.detach().numpy()
                    feature_rgb[k, iobj+1, :] = feat

        toa = -1  # time of accident, we don't have it in GTACrash
        # print('frame_level feature extraction finished.')

        # Resize bboxes so it aligns with the processing later
        detections[1] *= 1080
        detections[2] *= 720
        detections[3] *= 1080
        detections[4] *= 720
        #
        np.savez_compressed(
            out_file,
            feature=feature_rgb,
            flow_feat=feature_flow,
            detection=detections,
            vid_id=0,
            toa=toa
        )
        count +=1

if __name__ == '__main__':
    main()

# --path /mnt/experiments/sorlova/datasets/GTACrash/dataset
