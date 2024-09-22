import os
import numpy as np


np_path = "28_M.npz"

data = np.load(np_path)
vid_id = data["vid_id"]
features = data['feature']  # 100 x 31 x 2048
toa = [data['toa']+0]  # 1
detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
label = np.sum(detection[:, :, 5], axis=-1)
flow = data['flow_feat']  # 100 x 31 x 2048

deb = 0
