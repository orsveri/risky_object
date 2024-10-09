import sys
import os
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, repo_path)

import numpy as np
import pandas as pd
import cv2
import json
import zipfile
from tqdm import tqdm
import torchvision
import torch
from natsort import natsorted


IMG_EXT = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp')


def draw_bbox(img, bboxes, color):
    if np.sum(bboxes) == 0:
        print("No bboxes!")
        return img
    h, w, c = img.shape
    bs = bboxes.copy()
    bs[:, 0] = bboxes[:, 0] * w
    bs[:, 1] = bboxes[:, 1] * h
    bs[:, 2] = bboxes[:, 2] * w
    bs[:, 3] = bboxes[:, 3] * h
    bs = np.rint(bs).astype(int)
    for i, bbox in enumerate(bs):
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        img = cv2.putText(img, str(i), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def _find_non_conflicting_matches(S, s_threshold=0.3):
    # Get the maximum values in each row
    row_max = np.max(S, axis=1)
    # Get the maximum values in each column
    col_max = np.max(S, axis=0)
    # Find the indices where S[i, j] is the maximum in its row and column
    matches = []
    rows_left = list(range(S.shape[0]))
    cols_left = list(range(S.shape[1]))
    for i in range(S.shape[0]):  # i row
        for j in range(S.shape[1]):  # j column
            if S[i, j] < s_threshold:
                continue
            if S[i, j] == row_max[i] and S[i, j] == col_max[j]:
                matches.append((i, j))  # (row, column)
                if i in rows_left:
                    rows_left.pop(rows_left.index(i))
                if j in cols_left:
                    cols_left.pop(cols_left.index(j))
                break
    return matches, rows_left, cols_left


def _match_carla_iseg_bbox(track_bbox, det_bbox):
    xmin1, ymin1, xmax1, ymax1 = track_bbox
    xmin2, ymin2, xmax2, ymax2 = det_bbox
    # Intersection area
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    # Compute area of each box
    carla_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    iseg_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    carla_cx = 0.5 * (xmin1 + xmax1)
    carla_cy = 0.5 * (ymin1 + ymax1)
    iseg_cx = 0.5 * (xmin2 + xmax2)
    iseg_cy = 0.5 * (ymin2 + ymax2)
    distance = np.sqrt((carla_cx - iseg_cx) ** 2 + (carla_cy - iseg_cy) ** 2)
    # We have some rules!
    inter_iseg = intersection_area / iseg_area
    inter_carla = intersection_area / carla_area
    return inter_carla, inter_iseg, distance


def _fix_tall_bboxes(fixed_df, img_w, img_h):
    bboxes = fixed_df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    rates = np.nan_to_num(hs * img_h / ws / img_w)
    tall_indices = np.where(rates > 2.)[0].tolist()
    if len(tall_indices) == 0:
        return fixed_df
    for idx in tall_indices:
        if bboxes[idx, 0] < 4/710 or bboxes[idx, 2] > 706/710:
            continue
        w = ws[idx] * img_w
        new_ymin = bboxes[idx, 3] - (w / img_h)  # square
        assert new_ymin >= 0
        fixed_df.loc[idx, 'ymin'] = new_ymin
    return fixed_df


def _check_remaining(S, exact_matches, rows_left, cols_left, labels, track_bboxes, det_bboxes, img=None):
    other_matches = []
    not_ok_rows_left = rows_left.copy()
    for track_id, l in zip(rows_left, labels):
        i = not_ok_rows_left.index(track_id)
        tb = track_bboxes[track_id]
        area = (tb[2] - tb[0]) * (tb[3] - tb[1])
        # if there is one and only one _available_ detection inside
        inter_det_id = [idx for idx in np.where(S[track_id] > 0.)[0] if idx in cols_left]
        if len(inter_det_id) == 1:
            # if this detection is fully inside the track box
            inter_det_id = inter_det_id[0]
            db = det_bboxes[inter_det_id]
            if (db[0] >= tb[0]) and (db[1] >= tb[1]) and (db[2] <= tb[2]) and (db[3] <= tb[3]):
                cols_left.pop(cols_left.index(inter_det_id))
                not_ok_rows_left.pop(i)
                other_matches.append((track_id, inter_det_id))
                continue
        if area < 0.2 and tb[3] < 0.8:
            not_ok_rows_left.pop(i)
            continue
        if area > 0.8:  # typically, the bbox the size of the whole picture
            dets_left = det_bboxes[cols_left]
            possible_ids = np.where((dets_left[:, 1] == 0) | (dets_left[:, 3] >= 0.99))[0]
            if len(possible_ids) > 0:
                possible_areas = (dets_left[:, 2] - dets_left[:, 0]) * (dets_left[:, 3] - dets_left[:, 1])
                possible_heights = np.clip(dets_left[:, [1, 3]], a_min=0.3, a_max=0.9)
                possible_heights = (possible_heights[:, 1] - possible_heights[:, 0]) / 0.6
                possible_scores = possible_areas * possible_heights
                possible_scores = possible_scores[possible_ids]
                if np.max(possible_scores) > 0.1:
                    idx = possible_ids[np.argmax(possible_scores)]
                    det_id = cols_left.pop(cols_left[idx])
                    not_ok_rows_left.pop(i)
                    other_matches.append((track_id, det_id))
                    continue
        if area > 0.6 and l == 0:  # we can discard large bbox if it is not dangerous
            not_ok_rows_left.pop(i)
            continue
    # show image if it is given
    if img is not None:
        img = draw_bbox(img=img, bboxes=track_bboxes, color=(200, 140, 0))
        img = draw_bbox(img=img, bboxes=det_bboxes, color=(0, 200, 0))
        cv2.imshow(f"matches: {exact_matches}, not ok rows: {not_ok_rows_left}", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return other_matches, not_ok_rows_left


def match(track_bboxes, det_bboxes, track_labels, img=None):
    matches = []
    if track_bboxes.shape[0] == 0:
        return matches, True  # ok_frame = True
    det_inter_matrix = np.zeros(shape=(track_bboxes.shape[0], det_bboxes.shape[0]), dtype=float)
    track_inter_matrix = np.zeros(shape=(track_bboxes.shape[0], det_bboxes.shape[0]), dtype=float)
    center_distance_matrix = np.zeros(shape=(track_bboxes.shape[0], det_bboxes.shape[0]), dtype=float)
    carla_areas = np.zeros(shape=(track_bboxes.shape[0]), dtype=float)
    for ic, track_bbox in enumerate(track_bboxes):
        for ii, det_bbox in enumerate(det_bboxes):
            inter_track, inter_det, distance = _match_carla_iseg_bbox(track_bbox=track_bbox, det_bbox=det_bbox)
            det_inter_matrix[ic, ii] = inter_det
            track_inter_matrix[ic, ii] = inter_track
            center_distance_matrix[ic, ii] = distance
        carla_areas[ic] = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
    #det_inter_matrix = np.greater(det_inter_matrix, 0.9).astype(float)
    combined_matrix = det_inter_matrix * track_inter_matrix
    combined_matrix = np.nan_to_num(combined_matrix)
    center_distance_matrix = det_inter_matrix * center_distance_matrix
    # Find non-conflicting matches
    if combined_matrix.size == 0:
        return matches, True  # ok_frame = True
    # easy matches
    matches, rows_left, cols_left = _find_non_conflicting_matches(S=combined_matrix, s_threshold=0.15)
    if len(rows_left) == 0:
        # success
        return matches, True  # (track_idx, det_idx), ok_frame
    # other matches
    other_matches, not_ok_rows = _check_remaining(
        S=combined_matrix,
        exact_matches=matches,
        rows_left=rows_left,
        cols_left=cols_left,
        labels=track_labels,
        track_bboxes=track_bboxes,
        det_bboxes=det_bboxes,
        img=img
    )
    if other_matches is not None:
        matches = matches + other_matches
    if len(not_ok_rows) == 0:
        return matches, True
    # Otherwise
    return matches, False


def read_img_zip(zip_file, img_name):
    img = None
    with zipfile.ZipFile(zip_file, 'r') as z:
        with z.open(img_name) as file:
            image_data = file.read()
            np_img = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img


def fix_detections(track_file, det_file, output_det_file):
    # Iterate through all the files in the folder
    detections = []
    tracks = pd.read_csv(track_file)
    dets = pd.read_csv(det_file)
    fixed_df = tracks.copy(deep=True)

    rgb_zip = os.path.join(os.path.dirname(track_file), "rgb.zip")
    frames = natsorted(tracks['frame'].unique().tolist())
    if len(frames) == 0:
        return -1
    _img = read_img_zip(zip_file=rgb_zip, img_name=f"{str(frames[0]).zfill(6)}.jpg")
    fixed_df = _fix_tall_bboxes(fixed_df, img_w=_img.shape[1], img_h=_img.shape[0])

    for frame in frames:
        # dataset anno
        frame_tracks = fixed_df[fixed_df['frame'] == frame]
        frame_track_ids = frame_tracks.index.tolist()
        frame_tracks_labels = frame_tracks['label'].to_numpy().astype(int).tolist()
        frame_tracks_bboxes = frame_tracks[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        # yolo detections
        frame_dets = dets[(dets['frame'] == frame) & (dets['obj_class'].isin([2, 5, 7]))]  # car
        frame_dets_bboxes = frame_dets[["xmin", "ymin", "xmax", "ymax"]].to_numpy()

        # img = read_img_zip(zip_file=rgb_zip, img_name=f"{str(frame).zfill(6)}.jpg")
        # assert img is not None
        img = None

        matches, ok_frame = match(track_bboxes=frame_tracks_bboxes, track_labels=frame_tracks_labels, det_bboxes=frame_dets_bboxes, img=img)
        img = None
        if not ok_frame:
            if img is not None:
                img = draw_bbox(img=img, bboxes=frame_tracks_bboxes, color=(200, 140, 0))
                img = draw_bbox(img=img, bboxes=frame_dets_bboxes, color=(0, 200, 0))
                cv2.imshow(f"matches: {matches}, frame ok: {ok_frame}", img)
                cv2.waitKey()
                cv2.destroyAllWindows()
            return track_file
        for track_idx, det_idx in matches:
            correct_coords = frame_dets_bboxes[det_idx]
            track_index = frame_track_ids[track_idx]
            fixed_df.loc[track_index, 'xmin'] = correct_coords[0]
            fixed_df.loc[track_index, 'ymin'] = correct_coords[1]
            fixed_df.loc[track_index, 'xmax'] = correct_coords[2]
            fixed_df.loc[track_index, 'ymax'] = correct_coords[3]

    # save detections
    fixed_df.to_csv(output_det_file, header=True, index=True)
    return None


if __name__ == "__main__":
    input_root = '/mnt/experiments/sorlova/datasets/GTACrash/dataset'
    image_zip = '/mnt/experiments/sorlova/datasets/GTACrash/dataset/{}/rgb.zip'
    det_file = '/mnt/experiments/sorlova/datasets/GTACrash/dataset/{}/det_yolov5.csv'
    track_file = '/mnt/experiments/sorlova/datasets/GTACrash/dataset/{}/detections.csv'
    out_file = '/mnt/experiments/sorlova/datasets/GTACrash/dataset/{}/fixed_dets.csv'

    failed_clips = []
    fixed_clips = []
    no_frame_clips = []

    for clip in tqdm(natsorted(os.listdir(input_root))):
        clip_dir = os.path.join(input_root, clip)
        if not os.path.isdir(clip_dir):
            continue
        out_path = out_file.format(clip)
        # if os.path.exists(out_path):
        #     continue
        problem_track = fix_detections(
            track_file=track_file.format(clip),
            det_file=det_file.format(clip),
            output_det_file=out_path
        )
        if problem_track == -1:
            no_frame_clips.append(clip)
        elif problem_track is None:
            fixed_clips.append(clip)
        else:
            failed_clips.append(clip)

    print(f"fixed: {len(fixed_clips)}, incorrect: {len(failed_clips)}, no frames: {len(no_frame_clips)}")
    with open("gta_fixed.txt", 'w') as file:
        for item in fixed_clips:
            file.write(f"{item}\n")
    with open("gta_incorrect.txt", 'w') as file:
        for item in failed_clips:
            file.write(f"{item}\n")
    with open("gta_no_frame.txt", 'w') as file:
        for item in no_frame_clips:
            file.write(f"{item}\n")

