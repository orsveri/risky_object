from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch
import multiprocessing
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import DataLoader
from models.model import RiskyObject
from models.evaluation import evaluation, plot_auc_curve, plot_pr_curve, frame_auc
from dataloader import MyDataset
import argparse
from tqdm import tqdm
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import csv


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_class_weights(data_dir):
    if "Updated_feature" in data_dir:
        class_weights = [0.25, 1.]
    elif "AMNet_DoTA" in data_dir:
        class_weights = [0.16, 1.]
    elif "GTACrash/AMNet_GTA" in data_dir:
        class_weights = [0.21, 1.]
    else:
        raise NotImplementedError(f"Unknown class weights for this dataset! {data_dir}")
    return class_weights


def write_scalars(logger, epoch, losses, lr, obj_losses):
    # fetch results
    cross_entropy = losses['cross_entropy'].mean()
    # write to tensorboardX
    logger.add_scalar('train/loss', cross_entropy, epoch)
    logger.add_scalar('train/mean_loss', obj_losses, epoch)
    logger.add_scalar("train/lr", lr, epoch)


def write_test_scalars(logger, epoch, losses, roc_auc, ap):
    cross_entropy = losses.mean()
    logger.add_scalar('test/loss', cross_entropy, epoch)
    logger.add_scalar('test/roc_auc', roc_auc, epoch)
    logger.add_scalar('test/ap', ap, epoch)
    # logger.add_scalar('test/fpr', fpr, epoch)
    # logger.add_scalar('test/fpr', tpr, epoch)


def write_pr_curve_tensorboard(logger, test_probs, test_label):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    # tensorboard_truth = test_label == class_index
    tensorboard_truth = np.array(test_label)
    tensorboard_probs = np.array([test_probs[i][0] for i in range(len(test_probs))])

    # tensorboard_probs = test_probs[:, 1]
    # print(test_probs)
    # print(test_label[2])
    classes = ['no_risk', 'risk']

    logger.add_pr_curve(classes[1],
                        tensorboard_truth,
                        tensorboard_probs)
    # global_step=global_step)


def write_weight_histograms(logger, model, epoch):
    logger.add_histogram('histogram/gru.weight_ih_l0', model.gru_net.gru.weight_ih_l0, epoch)
    logger.add_histogram('histogram/gru.weight_hh_l0', model.gru_net.gru.weight_hh_l0, epoch)
    logger.add_histogram('histogram/gru.bias_ih_l0', model.gru_net.gru.bias_ih_l0, epoch)
    logger.add_histogram('histogram/gru.bias_hh_l0', model.gru_net.gru.bias_hh_l0, epoch)
    logger.add_histogram('histogram/gru.weight_ih_l1', model.gru_net.gru.weight_ih_l1, epoch)
    logger.add_histogram('histogram/gru.weight_hh_l1', model.gru_net.gru.weight_hh_l1, epoch)
    logger.add_histogram('histogram/gru.bias_ih_l1', model.gru_net.gru.bias_ih_l1, epoch)
    logger.add_histogram('histogram/gru.bias_hh_l1', model.gru_net.gru.bias_hh_l1, epoch)
    # fc_layers
    logger.add_histogram('histogram/gru.dense1.weight', model.gru_net.dense1.weight, epoch)
    logger.add_histogram('histogram/gru.dense1.bias', model.gru_net.dense1.bias, epoch)
    logger.add_histogram('histogram/gru.dense2.weight', model.gru_net.dense2.weight, epoch)
    logger.add_histogram('histogram/gru.dense2.bias', model.gru_net.dense2.bias, epoch)


def test_one_sample_one_model(model, batch_xs_, batch_det_, batch_toas_, batch_flow_):
    losses, all_outputs, labels = model(batch_xs_, batch_det_, batch_toas_, batch_flow_)

    frame_preds = []
    frame_labels = []
    obj_labels = []
    all_pred_ = []

    T = len(all_outputs)
    for t in range(T):
        frame = all_outputs[t]
        if len(frame) == 0:
            frame_preds.append(0)
            frame_labels.append(0)
            continue
        else:
            frame_scores = []
            for j in range(len(frame)):
                score = np.exp(frame[j][:, 1]) / np.sum(np.exp(frame[j]), axis=1)[0]
                all_pred_.append(score)
                frame_scores.append(score)
                obj_labels.append(int(labels[t][j] + 0))  # added zero to convert array to scalar
            frame_preds.append(max(frame_scores)[0])
            frame_labels.append(max(obj_labels))
    return losses, all_pred_, frame_preds, obj_labels, frame_labels


def test_all(testdata_loader, models, streams, device):
    # Here we know that the batch size is 0 and we do not have batch dimension in the model's outputs
    all_pred = [[] for _ in models]
    all_labels = [[] for _ in models]
    losses_all = [[] for _ in models]
    all_toa = [[] for _ in models]
    all_frame_pred = [[] for _ in models]
    all_frame_labels = [[] for _ in models]

    with torch.no_grad():
        for batch_xs, batch_det, batch_toas, batch_flow in tqdm(testdata_loader):
            batch_xs = batch_xs.to(device, non_blocking=True)
            batch_det = batch_det.to(device, non_blocking=True)
            batch_toas = batch_toas.to(device, non_blocking=True)
            batch_flow = batch_flow.to(device, non_blocking=True)
            batch_toas = batch_toas.cpu().detach()[:, 0].tolist()
            [all_toa[im].extend(batch_toas) for im, _ in enumerate(models)]  # !

            for ib in range(batch_xs.shape[0]):
                batch_xs_ = torch.unsqueeze(batch_xs[ib], dim=0)
                batch_det_ = torch.unsqueeze(batch_det[ib], dim=0)
                batch_toas_ = None  # torch.unsqueeze(batch_toas[ib], dim=0)
                batch_flow_ = torch.unsqueeze(batch_flow[ib], dim=0)

                # Parallel inference
                for im, (model, stream) in enumerate(zip(models, streams)):
                    with torch.cuda.stream(stream):
                        losses, all_pred_, frame_preds, obj_labels, frame_labels = test_one_sample_one_model(
                            model, batch_xs_, batch_det_, batch_toas_, batch_flow_
                        )
                        losses_all[im].append(losses)
                        all_pred[im].extend(all_pred_)
                        all_frame_pred[im].append(np.array(frame_preds, dtype=float))
                        all_frame_labels[im].append(np.array(frame_labels, dtype=float))
                        all_labels[im].append(obj_labels)
                # Sync point
                stream.synchronize()
    # all_frame_pred = np.array(all_frame_pred) changing seq length
    # all_pred = np.array([all_pred[i][0] for i in range(len(all_pred))])
    return losses_all, all_pred, all_labels, all_toa, all_frame_pred, all_frame_labels


def average_losses(losses_all):
    cross_entropy = 0
    for losses in losses_all:
        cross_entropy += losses['cross_entropy']
    losses_mean = cross_entropy/len(losses_all)
    return losses_mean


def _load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        # print('Checkpoint loaded')
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def test_eval():
    ckpt_dir = os.path.join(p.log_dir, "ckpts")
    assert os.path.exists(ckpt_dir)
    assert p.ckpt_start is not None and p.ckpt_end is not None
    epochs = list(range(p.ckpt_start, p.ckpt_end+1))
    model_paths = sorted([p.ckpt_file.format(ep) for ep in range(p.ckpt_start, p.ckpt_end+1)])
    assert all([os.path.exists(os.path.join(ckpt_dir, mp)) for mp in model_paths])
    assert len(model_paths) < 70

    result_dir = os.path.join(p.log_dir, "results", p.dataset)
    os.makedirs(result_dir, exist_ok=True)
    out_csv_path = os.path.join(result_dir, p.csv_file.format(p.dataset))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_path = dataset_paths[p.dataset]
    test_data = MyDataset(data_path, 'val', toTensor=True, device=device, data_fps=p.dfps, target_fps=p.tfps, n_clips=p.d)  # val
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size,
                                 shuffle=False, drop_last=False, num_workers = 2, pin_memory=True)
    n_frames = 100  # unnecessary

    models = []
    streams = []

    # ======= UPDATE HERE
    print("Loading models..")
    for model_file in tqdm(model_paths):
        model = RiskyObject(p.x_dim, p.h_dim, n_frames, p.tfps)
        model = model.to(device=device)
        model.eval()
        model, _, _ = _load_checkpoint(model, filename=os.path.join(ckpt_dir, model_file))
        models.append(model)
        streams.append(torch.cuda.Stream())
    print('Checkpoints loaded successfully')
    # ======================

    print('Predicting.........')

    csv_data = []

    losses_all_list, all_pred_list, all_labels_list, all_toa_list, all_frame_pred_list, all_frame_labels_list = test_all(
        testdata_loader, models, streams, device)

    for im, epoch in enumerate(epochs):
        losses_all, all_pred, all_labels, all_toa, all_frame_pred, all_frame_labels = losses_all_list[im], all_pred_list[im], all_labels_list[im], all_toa_list[im], all_frame_pred_list[im], all_frame_labels_list[im]
        loss_val = average_losses(losses_all)
        fpr, tpr, roc_auc, tta, frame_results = evaluation(
            all_pred, all_labels, None, fps=p.tfps, threshold=p.threshold, toa=all_toa, all_frame_pred=all_frame_pred,
            all_frame_labels=all_frame_labels
        )
        # SAVE results
        all_labels_flat = []
        for label in all_labels:
            all_labels_flat.extend(label)
        ap = average_precision_score(np.array(all_labels_flat), np.array(all_pred))
        ap = plot_pr_curve(all_labels, all_pred, epoch, base_logdir=result_dir, tag=None)
        plot_auc_curve(fpr, tpr, roc_auc, epoch, base_logdir=result_dir, tag=None)
        # !!!ADD!!! results in a .csv
        csv_data.append([epoch, ap, roc_auc, tta['mTTA'], datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    # Save new batch of metrics
    csv_data = pd.DataFrame(csv_data, columns=["epoch", "AP", "AUC", "TTA", "timestamp"])
    if os.path.exists(out_csv_path):
        # If the file exists, append the new data
        df_existing = pd.read_csv(out_csv_path, index_col=0)
        df_final = pd.concat([df_existing, csv_data], ignore_index=True)
    else:
        # If the file doesn't exist, the combined data is just the new data
        df_final = csv_data
    # Save the updated DataFrame to CSV
    df_final.reset_index(drop=True, inplace=True)
    df_final.to_csv(out_csv_path, index=True, header=True)

    return


dataset_paths = {
    "rol": "/mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature",
    "gta": "/mnt/experiments/sorlova/datasets/GTACrash/AMNet_GTA",
    "dota": "/mnt/experiments/sorlova/datasets/ROL/AMNet_DoTA"
}


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rol',
                        help='The desired dataset.')
    parser.add_argument('--d', type=int, default=None, help='Number of clips.')
    parser.add_argument('--tfps', type=int, default=20, help='Target FPS. Default: 20')
    parser.add_argument('--dfps', type=int, default=20, help='The FPS of data. Default: 20')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size. Default: 1')
    parser.add_argument('--seed', type=int, default=123,
                        help='The random seed. Default: 123')
    parser.add_argument('--ckpt_start', type=int, default=None,
                        help='The random seed. Default: None')
    parser.add_argument('--ckpt_end', type=int, default=None,
                        help='The random seed. Default: None')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Recall threshold for Precision and TTA. Default: 0.8')
    parser.add_argument('--h_dim', type=int, default=256,
                        help='hidden dimension of the gru. Default: 256')
    parser.add_argument('--x_dim', type=int, default=2048,
                        help='dimension of the resnet output. Default: 2048')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='The training log dir')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='The result output dir')
    parser.add_argument('--ckpt_file', type=str, default='model_{}.pth',
                        help='model file')
    parser.add_argument('--csv_file', type=str, default='test_{}.csv',
                        help='model file')

    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    test_eval()


"""
export CUDA_VISIBLE_DEVICES=1 && cd repos/TADTAA/risky_object/ && conda activate gg

--dataset rol --dfps 20 --tfps 10 --output_dir test_rol --log_dir logs/2_stage/only_dota/version_2 --ckpt_start 0 --ckpt_end 3 

"""
