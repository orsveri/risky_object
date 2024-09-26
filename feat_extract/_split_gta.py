import os
import random
import shutil
from tqdm import tqdm
random.seed(42)

inp_dir = "/mnt/experiments/sorlova/datasets/GTACrash/dataset"
out_dir = "/mnt/experiments/sorlova/datasets/GTACrash/npz"
out_dir_train = "/mnt/experiments/sorlova/datasets/GTACrash/train"
out_dir_val = "/mnt/experiments/sorlova/datasets/GTACrash/val"

# val_dirs = os.listdir(out_dir_val)
# train_dirs = os.listdir(out_dir_train)
# acc_val = [d for d in val_dirs if not d.startswith("noacc_")]
# noacc_val = [d for d in val_dirs if d.startswith("noacc_")]
# acc_train = [d for d in train_dirs if not d.startswith("noacc_")]
# noacc_train = [d for d in train_dirs if d.startswith("noacc_")]

clips = sorted([elem for elem in os.listdir(inp_dir) if os.path.isdir(os.path.join(inp_dir, elem))])

if False:
    for cl in tqdm(clips):
        np_path = os.path.join(inp_dir, cl, f"{cl}.npz")
        if os.path.exists(np_path):
            shutil.copy2(np_path, out_dir)
            if os.path.exists(os.path.join(out_dir, f"{cl}.npz")):
                os.remove(np_path)

if True:
    split = 0.9
    clips = sorted([elem for elem in os.listdir(out_dir) if os.path.splitext(os.path.join(out_dir, elem))[1] == ".npz"])
    random.shuffle(clips)
    acc_clips = [d for d in clips if not d.startswith("noacc_")]
    noacc_clips = [d for d in clips if d.startswith("noacc_")]
    # Acc
    La = len(acc_clips)
    train_len = int(La * split)
    train_acc = acc_clips[:train_len]
    val_acc = acc_clips[train_len:]
    # No acc
    Lna = len(noacc_clips)
    train_len = int(Lna * split)
    train_na = noacc_clips[:train_len]
    val_na = noacc_clips[train_len:]
    #
    for clip in tqdm(train_acc):
        shutil.copy2(os.path.join(out_dir, clip), out_dir_train)
    for clip in tqdm(val_acc):
        shutil.copy2(os.path.join(out_dir, clip), out_dir_val)
    for clip in tqdm(train_na):
        shutil.copy2(os.path.join(out_dir, clip), out_dir_train)
    for clip in tqdm(val_na):
        shutil.copy2(os.path.join(out_dir, clip), out_dir_val)
    print("")

if False:
    L = len(clips)
    train_len = int(L * 0.9)

    random.shuffle(clips)
    train_clips = clips[:train_len]
    val_clips = clips[train_len:]

    for clip in tqdm(train_clips):
        shutil.copy2(os.path.join(inp_dir, clip, f"{clip}.npz"), out_dir_train)

    for clip in tqdm(val_clips):
        shutil.copy2(os.path.join(inp_dir, clip, f"{clip}.npz"), out_dir_val)
