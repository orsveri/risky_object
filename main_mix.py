from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import date
import time

import torch
import multiprocessing
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import DataLoader, ConcatDataset
from models.model import RiskyObject
from models.evaluation import evaluation, plot_auc_curve, plot_pr_curve, frame_auc
from dataloader import MyDataset, ConcatBatchSampler
import argparse
from tqdm import tqdm
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import csv
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# optional

# seed = 123
# np.random.seed(seed)
# torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_class_weights(data_dir1, data_dir2):
    class_counts = []
    if "Updated_feature" in data_dir1 or "Updated_feature" in data_dir2:
        class_counts.append([282270, 69266])
    if "AMNet_DoTA" in data_dir1 or "AMNet_DoTA" in data_dir2:
        class_counts.append([133641, 21350])
    if "GTACrash/AMNet_GTA" in data_dir1 or "GTACrash/AMNet_GTA" in data_dir2:
        class_counts.append([451727, 88499])
    assert len(class_counts) == 2, f"Unknown class counts for this dataset! {data_dir1, data_dir2}"
    acc_counts = class_counts[0][1] + class_counts[1][1]
    noacc_counts = class_counts[0][0] + class_counts[1][0]
    csum = acc_counts + noacc_counts
    wacc = csum / acc_counts
    wnoacc = csum / noacc_counts
    wmax = max(wacc, wnoacc)
    class_weights = [wnoacc/wmax, wacc/wmax]
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


def test_all(testdata_loader, model, device):
    # Hre we know that the batch size is 0 and we do not have batch dimension in the model's outputs
    all_pred = []
    all_labels = []
    losses_all = []
    all_toa = []
    all_frame_pred = []
    all_frame_labels = []

    with torch.no_grad():
        for batch_xs, batch_det, batch_toas, batch_flow in tqdm(testdata_loader):
            batch_xs = batch_xs.to(device, non_blocking=True)
            batch_det = batch_det.to(device, non_blocking=True)
            batch_toas = batch_toas.to(device, non_blocking=True)
            batch_flow = batch_flow.to(device, non_blocking=True)
            all_toa.extend(batch_toas.cpu().detach()[:, 0].tolist())

            for ib in range(batch_xs.shape[0]):
                batch_xs_ = torch.unsqueeze(batch_xs[ib], dim=0)
                batch_det_ = torch.unsqueeze(batch_det[ib], dim=0)
                batch_toas_ = None  # torch.unsqueeze(batch_toas[ib], dim=0)
                batch_flow_ = torch.unsqueeze(batch_flow[ib], dim=0)

                losses, all_outputs, labels = model(batch_xs_, batch_det_, batch_toas_, batch_flow_)

                frame_preds = []
                frame_labels = []
                obj_labels = []

                losses_all.append(losses)
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
                            score = np.exp(frame[j][:, 1])/np.sum(np.exp(frame[j]), axis=1)[0]
                            all_pred.append(score)
                            frame_scores.append(score)
                            obj_labels.append(int(labels[t][j]+0))  # added zero to convert array to scalar
                        frame_preds.append(max(frame_scores)[0])
                        frame_labels.append(max(obj_labels))
                all_frame_pred.append(np.array(frame_preds, dtype=float))
                all_frame_labels.append(np.array(frame_labels, dtype=float))
                all_labels.append(obj_labels)
    #all_frame_pred = np.array(all_frame_pred) changing seq length
    # all_pred = np.array([all_pred[i][0] for i in range(len(all_pred))])
    return losses_all, all_pred, all_labels, all_toa, all_frame_pred, all_frame_labels


def average_losses(losses_all):
    cross_entropy = 0
    for losses in losses_all:
        cross_entropy += losses['cross_entropy']
    losses_mean = cross_entropy/len(losses_all)
    return losses_mean


def train_eval():
    # data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    model_dir = os.path.join(p.output_dir, 'ckpts')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_dir = os.path.join(p.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = SummaryWriter(logs_dir)

    today = date.today()
    date_saved = today.strftime("%b-%d-%Y")

    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)

    # optional
    result_dir = os.path.join(p.output_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #
    train_data1 = MyDataset(p.data_path1, 'train', toTensor=True, device=device, data_fps=p.dfps1, target_fps=p.tfps,
                            n_clips=p.d1, required_T=p.req_seq_len1)
    train_data2 = MyDataset(p.data_path2, 'train', toTensor=True, device=device, data_fps=p.dfps2, target_fps=p.tfps,
                            n_clips=p.d2, required_T=p.req_seq_len2)
    train_data = ConcatDataset([train_data1, train_data2])
    test_data1 = MyDataset(p.data_path1, 'val', toTensor=True, device=device, data_fps=p.dfps1, target_fps=p.tfps,
                           n_clips=p.d1, required_T=p.req_seq_len1)
    test_data2 = MyDataset(p.data_path2, 'val', toTensor=True, device=device, data_fps=p.dfps2, target_fps=p.tfps,
                           n_clips=p.d2, required_T=p.req_seq_len2)
    test_data = ConcatDataset([test_data1, test_data2])
    test_data = test_data
    #

    train_sampler = ConcatBatchSampler(dataset=train_data, batch_size=p.batch_size, shuffle=True)
    test_sampler = ConcatBatchSampler(dataset=test_data, batch_size=p.batch_size, shuffle=True)

    traindata_loader = DataLoader(dataset=train_data, batch_sampler=train_sampler, num_workers = 2, pin_memory=True)
    testdata_loader = DataLoader(dataset=test_data, batch_sampler=test_sampler, num_workers = 2, pin_memory=True)

    n_frames = 100
    model_file = p.ckpt_file
    class_weights = get_class_weights(p.data_path1, p.data_path2)
    model = RiskyObject(p.x_dim, p.h_dim, n_frames, p.tfps, class_weights=class_weights)

    result_csv = os.path.join(result_dir, f'result_{date_saved}_{current_time}.csv')
    with open(result_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"data_path1: {p.data_path1}, n_clips {p.d1}, dfps {p.dfps1}"])
        writer.writerow([f"data_path2: {p.data_path2}, n_clips {p.d1}, dfps {p.dfps1}"])
        writer.writerow(
            [f"x_dim: {model.x_dim}, base_h_dim: {model.h_dim}, base_n_layers: {model.n_layers}, "
             f"cor_n_layers: {model.n_layers_cor}, h_dim_cor:{model.h_dim_cor}, weight: {model.weight}, "
             f"tfps: {p.tfps}, seed: {p.seed}, batch_size: {p.batch_size}, ckpt: {p.ckpt_file}"])
        writer.writerow(['epoch', 'loss_val', 'mTTA', 'TTA_RT', 'P_RT', 'threshold', 'roc_auc', 'ap'])

    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    model = model.to(device=device)

    # for param in model.parameters():
    #     print(param)

    model.train()
    start_epoch = -1
    # resume training
    # -----------------
    if p.resume:
        model, optimizer, start_epoch = _load_checkpoint(model, optimizer = optimizer,  filename=model_file)
    # TO-DO:
    # -----------------
    # write histograms
    #write_weight_histograms(logger, model, 0)

    iter_cur = 0
    auc_max_list = [0, 0, 0]
    auc_max_k = [-1, -1, -1]
    ap_max_list = [0, 0, 0]
    ap_max_k = [-1, -1, -1]
    # optional
    # roc_auc = 0
    # ap = 0
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name)
    # sys.exit(1)

    if p.tl:
        for name, param in model.named_parameters():
            if 'dense1'in name or 'dense2' in name or 'soft_attention' in name or 'soft_attention_cor' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    for k in range(p.epoch):
        epoch_losses = []
        L_data = len(traindata_loader)
        loop = tqdm(enumerate(traindata_loader), total=L_data)
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (batch_xs, batch_det, batch_toas, batch_flow) in loop:
            batch_xs = batch_xs.to(device, non_blocking=True)
            batch_det = batch_det.to(device, non_blocking=True)
            batch_toas = batch_toas.to(device, non_blocking=True)
            batch_flow = batch_flow.to(device, non_blocking=True)

            for ib in range(batch_xs.shape[0]):
                batch_xs_ = torch.unsqueeze(batch_xs[ib], dim=0)
                batch_det_ = torch.unsqueeze(batch_det[ib], dim=0)
                batch_toas_ = torch.unsqueeze(batch_toas[ib], dim=0)
                batch_flow_ = torch.unsqueeze(batch_flow[ib], dim=0)

                optimizer.zero_grad()

                losses, all_outputs, all_labels = model(batch_xs_, batch_det_, batch_toas_, batch_flow_)

                losses_ = np.mean(np.array(losses['ce_list']))

                # backward
                losses['cross_entropy'].mean().backward()
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # not sure
                optimizer.step()
                loop.set_description(f"Epoch  [{k}/{p.epoch}]")
                loop.set_postfix(loss=losses['cross_entropy'].item())

                # -----------------
                # write scalars
                # To-DO:
                lr = optimizer.param_groups[0]['lr']
                write_scalars(logger, k, losses, lr, losses_)
                # Log gradients
                log_gradients = False  # 20-42 Mb per epoch
                if log_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            logger.add_histogram(f'{name}.grad', param.grad, k * L_data + i*p.batch_size+ib)  # Log the gradient histogram

                # ---------------
                iter_cur = 0

        # if k % p.test_iter == 0 and k != 0:
        print('----------------------------------')
        print("Starting evaluation...")
        model.eval()
        losses_all, all_pred, all_labels, all_toa, all_frame_pred, all_frame_labels = test_all(testdata_loader, model, device)

        loss_val = average_losses(losses_all)
        fpr, tpr, roc_auc, tta, frame_results = evaluation(
            all_pred, all_labels, k, fps=p.tfps, threshold=p.threshold, toa=all_toa, all_frame_pred=all_frame_pred,
            all_frame_labels=all_frame_labels
        )
        plot_auc_curve(fpr, tpr, roc_auc, k)
        ap = plot_pr_curve(all_labels, all_pred, k)

        print(f"AUC : {roc_auc:.4f}")
        print(f"AP : {ap:.4f}")
        # print('testing loss :', loss_val)
        # keep track of validation losses
        write_test_scalars(logger, k, loss_val, roc_auc, ap)

        with open(result_csv, 'a+', newline='') as saving_result:
            writer = csv.writer(saving_result)
            writer.writerow([k, loss_val, tta["mTTA"], tta["TTA_RT"], tta["P_RT"], tta["threshold"], roc_auc, ap])

        model.train()

        all_labels_flat = []
        for label in all_labels:
            all_labels_flat.extend(label)
        write_pr_curve_tensorboard(logger, all_pred, all_labels_flat)

        # # save model
        save_only_best = False
        if save_only_best:
            auc_max_list_ = auc_max_list + [roc_auc]
            del_k = np.argmin(auc_max_list_)
            if del_k != 3:
                del_path = os.path.join(model_dir, f'best_auc_{auc_max_k[del_k]}.pth')
                if os.path.exists(del_path):
                    os.remove(del_path)
                model_file = os.path.join(model_dir, f'best_auc_{k}.pth')
                torch.save({'epoch': k,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, model_file)
                auc_max_list[del_k] = roc_auc
                auc_max_k[del_k] = k
                print(f'Best AUC Model has been saved as: {model_file}. (del {auc_max_k[del_k]}, auc_list {auc_max_list})')
            ap_max_list_ = ap_max_list + [ap]
            del_k = np.argmin(ap_max_list_)
            if del_k != 3:
                del_path = os.path.join(model_dir, f'best_ap_{ap_max_k[del_k]}.pth')
                if os.path.exists(del_path):
                    os.remove(del_path)
                model_file = os.path.join(model_dir, f'best_ap_{k}.pth')
                torch.save({'epoch': k,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, model_file)
                ap_max_list[del_k] = ap
                ap_max_k[del_k] = k
                print(f'Best AP Model has been saved as: {model_file}. (del {ap_max_k[del_k]}, ap_list {ap_max_list})')
        else:
            model_file = os.path.join(model_dir, f'model_{k}.pth')
            torch.save({'epoch': k,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, model_file)
            print(f'Model has been saved as: {model_file}.')
        ###########################################################################
        scheduler.step(losses['cross_entropy'])
        # write histograms
        #write_weight_histograms(logger, model, k+1)


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



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path1', type=str,
                        default='/mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature',
                        # '/mnt/experiments/sorlova/datasets/ROL/AMNet_DoTA/'
                        help='The relative path of dataset.')
    parser.add_argument('--data_path2', type=str,
                        default='/mnt/experiments/sorlova/datasets/GTACrash/AMNet_feats',
                        # '/mnt/experiments/sorlova/datasets/ROL/AMNet_DoTA/'
                        help='The relative path of dataset.')
    parser.add_argument('--d1', type=int, default=None, help='Number of clips.')
    parser.add_argument('--d2', type=int, default=None, help='Number of clips.')
    parser.add_argument('--tfps', type=int, default=20, help='Target FPS. Default: 20')
    parser.add_argument('--dfps1', type=int, default=20, help='The FPS of data. Default: 20')
    parser.add_argument('--dfps2', type=int, default=10, help='The FPS of data. Default: 20')
    parser.add_argument('--req_seq_len1', type=int, default=None, help='Sequence (video) length')
    parser.add_argument('--req_seq_len2', type=int, default=None, help='Sequence (video) length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--seed', type=int, default=123,
                        help='The random seed. Default: 1')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Recall threshold for Precision and TTA. Default: 0.8')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=30,
                        help='The number of training epoches. Default: 30')
    parser.add_argument('--h_dim', type=int, default=256,
                        help='hidden dimension of the gru. Default: 256')
    parser.add_argument('--x_dim', type=int, default=2048,
                        help='dimension of the resnet output. Default: 2048')
    parser.add_argument('--phase', type=str, default='train', choices=['check', 'train', 'test'],
                        help='dimension of the resnet output. Default: 2048')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='The log dir')
    parser.add_argument('--test_iter', type=int, default=1,
                        help='The number of epochs to perform a evaluation process. Default: 64')
    parser.add_argument('--ckpt_file', type=str, default='',  # 'checkpoints/pretrained/best_auc.pth'
                        help='model file')
    parser.add_argument('--resume', action='store_true',
                        help='If to resume the training. Default: False')
    parser.add_argument('--tl', action='store_true',
                        help='If want transfer learning. Default: False')

    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    assert p.phase == 'train'
    train_eval()

# --data_path /mnt/experiments/sorlova/datasets/Updated_feature/Updated_feature/ --phase=train --batch_size=1 --output_dir=logs/check2 --dfps 20 --tfps 20

"""
export CUDA_VISIBLE_DEVICES=0 && cd repos/TADTAA/risky_object/ && conda activate gg

python main_mix.py --data_path1 /mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature/ --dfps1 20 --data_path2 /mnt/experiments/sorlova/datasets/GTACrash/AMNet_GTA --dfps2 10 --req_seq_len2 19 --d2 2000 --phase=train --epoch 60 --tfps 10 --batch_size=32 --output_dir=logs/2_stage/rol_gta_2000/version_1 --seed 123
python main_mix.py --data_path1 /mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature/ --dfps1 20 --data_path2 /mnt/experiments/sorlova/datasets/GTACrash/AMNet_GTA --dfps2 10 --req_seq_len2 19 --d2 2000 --phase=train --epoch 60 --tfps 10 --batch_size=32 --output_dir=logs/2_stage/rol_gta_2000/version_2 --seed 42 
python main_mix.py --data_path1 /mnt/experiments/sorlova/datasets/ROL/Updated_feature/Updated_feature/ --dfps1 20 --data_path2 /mnt/experiments/sorlova/datasets/GTACrash/AMNet_GTA --dfps2 10 --req_seq_len2 19 --d2 2000 --phase=train --epoch 60 --tfps 10 --batch_size=32 --output_dir=logs/2_stage/rol_gta_2000/version_3 --seed 256


"""
