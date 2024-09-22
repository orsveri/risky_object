# Data preprocessing

(Mostly derived from [here](https://github.com/monjurulkarim/risky_object/issues/12))

There are two parts to the dataset suitable for AM-Net:

- Input RGB video
- Annotations

The preprocessing steps for each part are described below.

## Input RGB video

From RGB video, we will generate Optical Flow, detections and frame features.

! Detections (and probably all the other features) are generated for video frame size (1080, 720).

1. [`TODO: script`] Take input RGB video and generate Optical flow videos using [RAFT](https://github.com/princeton-vl/RAFT). After this, processing of RGB and Flow videos are mostly identical. Authors clipped videos so each clip contains 5 s of action, or 100 frames with 20 FPS.

1. AM-Net expects tracked detections to contain 6 valies for each bounding box: \
`[track_id, x1, y1, x2, y2, label{0,1}]`. It is done by using YOLOv5 and DeepSort with frame size of (1080, 720).
   1. [`TODO: script`] Run YOLOv5 to get detections: TODO: script 
   1. [`TODO: script`] Run DeepSort to associate detections: TODO: script

1. [`feat_extract/feat_extract.py`] For each clip containing N frames, generate **RGB appearance feature** with shape (N, 31, 2048) which consists of: 
   - **frame-level feature** with shape (N, 1, 2048)
   - **box-level features** with shape (N, 30, 2048) \
*RGB appearance features are not really used for training and validation.*

1. [`feat_extract/feat_extract.py`] For each clip containing N frames, generate **Flow appearance feature** using the same code.

1. Combine all the data into .npz files

## Annotations

Actually, "time of accident" - `toa` in the code, is not really used. So only object-level collision labels are used, which are part of the bounding box annotations.

## CARLA data

Bounding boxes and actor ids are provided 

