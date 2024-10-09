import os
import numpy as np
import cv2
import zipfile
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from natsort import natsorted


def get_obj_labels(npz_file, filter_empty_frames=False):
    data = np.load(npz_file)
    detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
    frame_detections = []
    for frame in range(detection.shape[0]):
        actual_detection = detection[frame, detection[frame, :, 0] > 0]
        frame_detections.append(actual_detection)
    return frame_detections


def draw_bbox(img, dets):
    tracks = dets[:, 0]
    bboxes = dets[:, 1:5].copy()
    labels = dets[:, 5]
    if np.sum(bboxes) == 0:
        return img
    h, w, c = img.shape
    bboxes[:, 0] = bboxes[:, 0] / 1080 * w
    bboxes[:, 1] = bboxes[:, 1] / 720 * h
    bboxes[:, 2] = bboxes[:, 2] / 1080 * w
    bboxes[:, 3] = bboxes[:, 3] / 720 * h
    bboxes = np.rint(bboxes).astype(int)
    for bbox, track, label in zip(bboxes, tracks, labels):
        color = (0, 0, 200) if label else (0, 200, 0)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img, str(int(track)), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def show_boxes(img_dir, anno_file, out_dir):
    detections = get_obj_labels(anno_file)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(img_dir, 'r') as z:
        images = natsorted([file_info.filename for file_info in z.infolist() if file_info.filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        if len(detections) - len(images) == 1:
            images = images[1:]
        for img_filename, dets in tqdm(zip(images, detections)):
            out_path = os.path.join(out_dir, img_filename)
            with z.open(img_filename) as file:
                image_data = file.read()
                np_img = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                img = draw_bbox(img, dets)
                cv2.imwrite(os.path.join(out_path), img)


img_dir = "/mnt/experiments/sorlova/datasets/GTACrash/dataset/000000/rgb.zip"
anno_file = "/mnt/experiments/sorlova/datasets/GTACrash/AMNet_feats/train/000000.npz"
out_dir = "feat_extract/gta_example/000000"

show_boxes(
    img_dir=img_dir,
    anno_file=anno_file,
    out_dir = out_dir
)

