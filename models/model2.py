from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import inspect
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
import torch.nn.functional as F
import numpy as np
import sys


def update_track_ids(features):
    # Initialize parameters
    L, N, C = features.shape
    new_track_ids = []
    next_id = 10  # Next available track ID
    active_tracks = {}  # Dictionary to track active objects
    new_ids = []

    for frame_idx in range(L):
        current_frame_ids = features[frame_idx, :, 0].cpu().numpy()  # Get the object IDs for the current frame
        new_frame_ids = []

        for obj_idx in range(N):
            current_id = current_frame_ids[obj_idx]

            if current_id == 0:  # Assuming 0 means no object detected
                new_frame_ids.append(0)  # No object present
                active_tracks.pop(current_id, None)  # We do not observe this object anymore
                continue

            if current_id not in active_tracks:
                # New object detected, assign a new track ID
                active_tracks[current_id] = next_id
                next_id += 1

            new_frame_ids.append(active_tracks[current_id])  # Assign the new track ID

        new_ids.append(new_frame_ids)

    # Update the features tensor with the new IDs
    features[:, :, 0] = torch.tensor(np.array(new_ids)).to(features.device)

    return features


def break_video_tensor(A):
    # tensor A with shape (T, N, C)
    l = A[:, :, 0]  # Shape (100, 30)
    C = A.shape[-1]
    unique_values = torch.unique(l)

    # Step 1: Create a list to store separate tensors
    separated_tensors = []
    masks = []
    for value in unique_values:
        if value.item() == 0.:
            continue
        # Step 2: Use boolean indexing to create a mask
        mask = (l == value)
        # Step 3: Extract the corresponding entries from A
        # This requires handling the dimensions correctly
        # Using unsqueeze to match the dimensions of A for boolean indexing
        extracted_tensor = A[mask].view(-1, C)  # Shape will be (num_selected, C)
        masks.append(mask)

        # Store the extracted tensor in the dictionary
        separated_tensors.append(extracted_tensor)

    return separated_tensors, masks


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, output_cor_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = [0, 0]
        self.output_cor_dim = output_cor_dim
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        self.dense1 = torch.nn.Linear(hidden_dim+output_cor_dim, 256)
        self.dense2 = torch.nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, output_cor):
        out, h = self.gru(x, h)
        out = unpack_sequence(out)
        out = torch.cat(out, dim=-2)
        output_cor = unpack_sequence(output_cor)
        output_cor = torch.cat(output_cor, dim=-2)
        out = torch.cat([out, output_cor], dim=-1)
        out = F.dropout(out, self.dropout[0])  # optional
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        return out, h


class CorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(CorGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class flow_GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(flow_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class SpatialAttention(nn.Module):
    """
    Applied soft attention on the hidden representation of all the objects in a frame.
    """

    def __init__(self, h_dim):
        super(SpatialAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_dim, 1))  # hidden representation dimension is 256
        self.softmax = nn.Softmax(dim=1)
        import math
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, h_all_in):
        """
        :h_all_in - dictionary containing object tracking id and hidden representation of size 2 x 1 x 256 each
        :output - dictionary with the same shape of h_all_in
        """
        k = []
        v = []
        for key in h_all_in:
            v.append(h_all_in[key])
            k.append(key)

        if len(v) != 0:
            h_in = torch.cat([element for element in v], dim=1)
            m = torch.tanh(h_in)
            alpha = torch.softmax(torch.matmul(m, self.weight), 1)
            roh = torch.mul(h_in, alpha)
            list_roh = []
            for i in range(roh.size(1)):
                list_roh.append(roh[:, i, :].unsqueeze(1).contiguous())

            h_all_in = {}
            for ke, value in zip(k, list_roh):
                h_all_in[ke] = value

        return h_all_in


class RiskyObject(nn.Module):
    def __init__(self, x_dim, h_dim, n_frames=100, fps=20.0):
        super(RiskyObject, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.fps = fps
        self.n_frames = n_frames
        self.n_layers = 2
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())  # rgb

        # for secondary GRU
        self.n_layers_cor = 1
        self.h_dim_cor = 32
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, self.n_layers, self.h_dim_cor)
        self.weight = torch.Tensor([0.25, 1]).cuda()  # TO-DO: find the correct weight

        # input dim 4
        self.gru_net_cor = CorGRU(4, self.h_dim_cor, self.n_layers_cor)
        self.soft_attention = SpatialAttention(h_dim)
        self.soft_attention_cor = SpatialAttention(self.h_dim_cor)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='mean')

    def forward(self, x, y, toa, flow, hidden_in=None, testing=False):
        """
        :param x (batchsize, nFrames, 1+maxBox, Xdim) (RGB appearance feats - not really used)
        :param y (batchsize, nFrames, maxBox, 6) (detections)
        :param toa (batchsize, 1) (time of accident? number of frame)
        :param flow (batchsize, nFrames, 1+maxBox, Xdim) (Flow appearance feats)
        :batchsize = 1, currently we support batchsize=1
        """
        losses = {'cross_entropy': 0}
        batch_size = 1
        maxBox = 30  # Max number of objects per frame
        hidden_size = self.h_dim
        coord_hidden_size = self.h_dim_cor

        # Prepare features
        x_val = self.phi_x(flow[0])  # (100 x 31 x 256)
        img_embed = x_val[:, 0, :].unsqueeze(1).repeat(1, maxBox, 1)  # (100 x 30 x 256)
        obj_embed = x_val[:, 1:, :]  # (100 x 30 x 256)
        x_val = torch.cat([obj_embed, img_embed], dim=-1)  # (100 x 30 x 512)

        # Normalize coords
        y[:, :, :, 1] /= 1280
        y[:, :, :, 2] /= 720
        y[:, :, :, 3] /= 1280
        y[:, :, :, 4] /= 720

        # Obtain object sequences
        y = update_track_ids(y[0])
        feat = torch.cat([y, x_val], dim=-1)  # (100 x 30 x 512)
        obj_sequences, obj_seq_masks = break_video_tensor(feat)

        # Split back
        feat_sequences = []
        coord_sequences = []
        labels_sequences = []
        len_sequences = []
        for obj_seq in obj_sequences:
            feat_sequences.append(obj_seq[:, 6:])
            coord_sequences.append(obj_seq[:, 1:5])
            labels_sequences.append(obj_seq[:, 5])
            len_sequences.append(obj_seq.shape[0])

        # Pack sequences
        packed_coord = pack_sequence(coord_sequences, enforce_sorted=False)
        packed_feat = pack_sequence(feat_sequences, enforce_sorted=False)

        # Initialize hidden states
        h_out = None  # torch.zeros(self.n_layers, batch_size, hidden_size).to(x.device)
        h_out_cor = None  #  torch.zeros(self.n_layers_cor, batch_size, coord_hidden_size).to(x.device)

        # Process with GRU
        packed_output_cor, h_out_cor = self.gru_net_cor(packed_coord, h_out_cor)
        unpacked_output, h_out = self.gru_net(packed_feat, h_out, packed_output_cor)

        # Recover the initial shape
        output = torch.split(unpacked_output, len_sequences)

        all_outputs = []
        all_labels = []

        all_outputs.append(frame_outputs)
        all_labels.append(frame_labels)

        # attention - here we have the problem!
        h_all_in = self.soft_attention(h_all_in)
        h_all_in_cor = self.soft_attention_cor(h_all_in_cor)

        return losses, all_outputs, all_labels
