# import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import os
import numpy as np


def evaluation_tta(all_frame_pred, all_frame_labels, time_of_accidents, threshold, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """
    all_labels_ = [max(sublist) for sublist in all_frame_labels]
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels_[idx] > 0:
            pred = all_frame_pred[idx][:int(toa)]  # positive video
        else:
            pred = all_frame_pred[idx]  # negative video
        # find the minimum prediction
        try:
            min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        except:
            min_pred = 0
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_frame_pred[0].shape[0] / fps

    # iterate a set of thresholds from the minimum predictions
    Precision = np.zeros((n_frames))
#     print(Precision.shape)
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
#     Precision = np.zeros((200))
#     Recall = np.zeros((200))
#     Time = np.zeros((200))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp = np.where(preds_eval[i]*all_labels_[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
        try:
            if Tp_Fp == 0:  # predictions of all videos are negative
                continue
            else:
                Precision[cnt] = Tp/Tp_Fp
            if np.sum(all_labels_) == 0:  # gt of all videos are negative
                continue
            else:
                Recall[cnt] = Tp/np.sum(all_labels_)
            if counter == 0:
                continue
            else:
                Time[cnt] = (1-time/counter)
            cnt += 1
        except:
            break
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _, rep_index = np.unique(Recall, return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    # print("Average Precision= %.4f, mean Time to accident= %.4f" % (AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
#     print(sort_recall)
    a = np.where(new_Recall >= 0)
    P_RT = new_Precision[a[0][0]]
    TTA_RT = sort_time[np.argmin(np.abs(sort_recall-threshold))] * total_seconds
    return {"mTTA": mTTA, "TTA_RT": TTA_RT, "P_RT": P_RT, "threshold": threshold}


def evaluation(all_pred, all_labels, epoch, fps, toa=None, all_frame_pred=None, threshold=0.3):
    all_labels_flat = []
    for label in all_labels:
        all_labels_flat.extend(label)
    fpr, tpr, thresholds = metrics.roc_curve(np.array(all_labels_flat), np.array(all_pred), pos_label=1)
    # np.savez('auc.npz', fpr=fpr, tpr=tpr, thresholds=thresholds)
    roc_auc = metrics.auc(fpr, tpr)
    tta = None
    if toa is not None and all_frame_pred is not None:
        tta = evaluation_tta(all_frame_pred=all_frame_pred, all_frame_labels=all_labels, time_of_accidents=toa, fps=fps, threshold=threshold)
    return fpr, tpr, roc_auc, tta


def plot_auc_curve(fpr, tpr, roc_auc, epoch, base_logdir=None, tag=None):
    curve_dir = 'charts/' if tag is None else f"{tag}_charts"
    if base_logdir is not None:
        curve_dir = os.path.join(base_logdir, curve_dir)
    if not os.path.exists(curve_dir):
        os.makedirs(curve_dir)
    auc_curve_file = os.path.join(curve_dir, 'auc_%02d.png' % (epoch))

    plt.title(f'Receiver Operating Characteristic at epoch: {epoch}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(auc_curve_file)
    plt.close()


def plot_pr_curve(all_labels, all_pred, epoch, base_logdir=None, tag=None):
    all_labels_flat = []
    for label in all_labels:
        all_labels_flat.extend(label)

    pr_dir = 'charts/' if tag is None else f"{tag}_charts"
    if base_logdir is not None:
        pr_dir = os.path.join(base_logdir, pr_dir)
    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)
    pr_curve_file = os.path.join(pr_dir, 'pr_%02d.png' % (epoch))
    precision, recall, thresholds = precision_recall_curve(np.array(all_labels_flat), np.array(all_pred))
    # np.savez('ap_attention_bbox_flow.npz', precision=precision,
    #          recall=recall, thresholds=thresholds)
    ap = average_precision_score(np.array(all_labels_flat), np.array(all_pred))

    plt.title(f'Precision-Recall Curve at epoch: {epoch}')
    plt.plot(recall, precision, 'b', label='AP = %0.2f' % ap)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(pr_curve_file)
    plt.close()
    return ap


def frame_auc(output, labels):
    # print(output)
    output = np.array(output)
    labels = np.array(labels)
    # print(output)
    all_pred = []
    all_labels = []

    for t in range(len(output)):
        frame = output[t]
        frame_score = []
        frame_label = []
        print(frame)

        if len(frame) == 0:
            continue
        else:
            for j in range(len(frame)):
                score = np.exp(frame[j][:, 1])/np.sum(np.exp(frame[j]), axis=1)
                frame_score.append(score)
                frame_label.append(labels[t][j]+0)
            all_pred.append(max(frame_score))
            all_labels.append(sum(frame_label))

    new_labels = []
    for i in all_labels:
        if i > 0.0:
            new_labels.append(1.0)
        else:
            new_labels.append(0.0)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(new_labels), np.array(all_pred), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc


# def evaluation(all_pred,all_labels):
#     # cm = confusion_matrix(all_labels, all_pred, labels=['no-risk','risk'])
#     TPs = 0
#     TNs = 0
#     FPs = 0
#     FNs = 0
#     for pred,gt in zip(all_pred,all_labels):
#         if gt ==0:
#             if pred ==0:
#                 TNs+=1
#             elif pred==1:
#                 FPs+=1
#         elif gt ==1:
#             if pred==0:
#                 FNs+=1
#             elif pred ==1:
#                 TPs+=1
#
#     cm = ([TNs,FPs],[FNs,TPs])
#     if TPs == 0:
#         recall =0
#         precision = 0
#         accuracy = (TPs+TNs)/(TPs+TNs+FPs+FNs)
#     else:
#         recall = TPs/(TPs+FNs)
#         precision = TPs/(TPs+FPs)
#         accuracy = (TPs+TNs)/(TPs+TNs+FPs+FNs)
#     return cm, precision, recall, accuracy
