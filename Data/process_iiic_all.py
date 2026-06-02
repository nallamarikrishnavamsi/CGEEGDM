import numpy as np
import mne
import pickle
import os
from tqdm import tqdm

np.random.seed(0)

# ch order not specified but mmost likely to be
ch_order = [  
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

# IMPORTANT already / 100 here so dont divide again in your dataloader
X = np.load("all_train_X.npy").astype(float) / 100 # ~0.6% |x| > 1
Y = np.load("all_train_Y2_hard.npy").argmax(axis=1).astype(int)
# print((np.load("all_train_Y.npy").sum(axis=1)> 3).sum())
# print((np.load("all_train_Y.npy").sum(axis=1)> 0).sum())
# exit()
x_subject = np.array([s.split("_")[0] for s in np.load("all_train_key.npy")])
subjects = np.unique(sorted(x_subject))
np.random.shuffle(subjects)

n_sub = subjects.shape[0]
# print(n_sub)
# exit()

train_sub_last = int(n_sub * 0.6) + 1
val_sub_last = train_sub_last + int(n_sub * 0.2) + 1

train_sub = subjects[:train_sub_last]
val_sub = subjects[train_sub_last:val_sub_last]
test_sub = subjects[val_sub_last:]
print("train", train_sub.tolist(), "val", val_sub.tolist(), "test", test_sub.tolist(), sep="\n")

# exit()

n_data = Y.shape[0]
train_xy = [[X[i], Y[i]] for i in range(n_data) if x_subject[i] in train_sub]
val_xy = [[X[i], Y[i]] for i in range(n_data) if x_subject[i] in val_sub]
test_xy = [[X[i], Y[i]] for i in range(n_data) if x_subject[i] in test_sub]

mne.set_log_level("CRITICAL")

if not os.path.exists("all/train"): os.mkdir("all/train")
i = 0
for x, y in tqdm(train_xy):
    with open(f"all/train/{i}.pkl", "wb") as f:
        pickle.dump({"signal": mne.filter.filter_data(x, 200, 0.1, 75), "label": y}, f)
    i += 1

if not os.path.exists("all/val"): os.mkdir("all/val")
i = 0
for x, y in tqdm(val_xy):
    with open(f"all/val/{i}.pkl", "wb") as f:
        pickle.dump({"signal": mne.filter.filter_data(x, 200, 0.1, 75), "label": y}, f)
    i += 1

if not os.path.exists("all/test"): os.mkdir("all/test")
i = 0
for x, y in tqdm(test_xy):
    with open(f"all/test/{i}.pkl", "wb") as f:
        pickle.dump({"signal": mne.filter.filter_data(x, 200, 0.1, 75), "label": y}, f)
    i += 1
