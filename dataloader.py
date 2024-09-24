from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, Sampler, ConcatDataset


class ConcatBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, dataset_sizes, shuffle=True):
        """
        Args:
            dataset (ConcatDataset): The concatenated dataset.
            batch_size (int): Number of samples per batch.
            dataset_sizes (list): A list containing the sizes of each dataset in the ConcatDataset.
            shuffle (bool): Whether to shuffle the indices within each dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.shuffle = shuffle

        # Create lists of indices for each dataset
        self.indices_a = list(range(self.dataset_sizes[0]))  # Indices for Dataset A
        self.indices_b = list(range(self.dataset_sizes[0], sum(self.dataset_sizes)))  # Indices for Dataset B

    def __iter__(self):
        # Optionally shuffle the indices within each dataset
        if self.shuffle:
            torch.manual_seed(torch.initial_seed())
            self.indices_a = torch.randperm(self.dataset_sizes[0]).tolist()
            self.indices_b = torch.randperm(self.dataset_sizes[1]).add(self.dataset_sizes[0]).tolist()

        # Split indices into batches for Dataset A and Dataset B
        batches_a = [self.indices_a[i:i + self.batch_size] for i in range(0, len(self.indices_a), self.batch_size)]
        batches_b = [self.indices_b[i:i + self.batch_size] for i in range(0, len(self.indices_b), self.batch_size)]

        if len(batches_a) > len(batches_b):
            na = round(len(batches_a) / len(batches_b))
            nb = 1
        else:
            na = 1
            nb = round(len(batches_b) / len(batches_a))

        # Interleave batches from both datasets (or alternate in some way)
        combined_batches = []
        while len(batches_a) and len(batches_b):
            _a_batches = batches_a[:na]
            _b_batches = batches_b[:nb]
            combined_batches.extend(_a_batches)
            combined_batches.append(_b_batches)
            batches_a = batches_a[na:]
            batches_b = batches_b[nb:]

        # Handle if one dataset has more batches than the other
        if len(batches_a) > 0:
            combined_batches.extend(batches_a)
        elif len(batches_b) > 0:
            combined_batches.extend(batches_b)

        # Return iterator over combined batches
        return iter(combined_batches)

    def __len__(self):
        # Total number of batches, considering both datasets
        num_batches_a = len(self.indices_a) // self.batch_size
        num_batches_b = len(self.indices_b) // self.batch_size
        return num_batches_a + num_batches_b


class MyDataset(Dataset):
    def __init__(self, data_path, phase, data_fps, target_fps, toTensor=False,  device=torch.device('cuda')):
        self.data_path = data_path
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.n_frames = 100  # -->
        self.fps = 20
        self.data_fps = data_fps
        self.target_fps = target_fps
        assert data_fps % target_fps == 0, "Cannot use the given target fps!"
        self.fps_step = int(data_fps // target_fps)
        self.dim_feature = 2048
        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)
        print(f"Dataset for {phase} initialized!")
        # print(self.files_list)

    def __len__(self):
        data_len = len(self.files_list)
        #return data_len
        return 10

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s" % (filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['feature']  # 100 x 31 x 2048
            toa = [data['toa']+0]  # 1
            detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
            flow = data['flow_feat']  # 100 x 31 x 2048
            # track_id : data['detection'][:,:,0] --> 100 x 30
        except:
            raise IOError('Load data error! File: %s' % (data_file))

        N = features.shape[0]
        if self.fps_step != 1:
            start = (N - 1) % self.fps_step
            features = features[start::self.fps_step]
            detection = detection[start::self.fps_step]
            flow = flow[start::self.fps_step]
            toa = [toa[0] / self.fps_step]

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)  # 50 x 20 x 4096
            detection = torch.Tensor(detection).to(self.device)
            toa = torch.Tensor(toa).to(self.device)
            flow = torch.Tensor(flow).to(self.device)

        return features, detection, toa, flow


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 10')

    p = parser.parse_args()
    # data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    data_path = p.data_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = MyDataset(data_path, 'train', toTensor=True, device=device, data_fps=20, target_fps=20)
    test_data = MyDataset(data_path, 'val', toTensor=True, device=device, data_fps=20, target_fps=20)

    traindata_loader = DataLoader(
        dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(
        dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    print('===Checking the dataloader====')
    for e in range(1):
        print('Epoch: %d' % (e))
        for i, (batch_xs, batch_det, batch_toas, batch_flow) in tqdm(enumerate(traindata_loader), total=len(traindata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('detection dim:', batch_det.size())
                print('toas dim:', batch_toas.size())
                print('flow dim : ', batch_flow.size())

    for e in range(1):
        print('Epoch: %d' % (e))
        for i, (batch_xs, batch_det, batch_toas, batch_flow) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('detection dim:', batch_det.size())
                print('toas dim:', batch_toas.size())
                print('flow dim : ', batch_flow.size())
