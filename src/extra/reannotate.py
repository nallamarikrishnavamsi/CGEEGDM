import numpy as np
import mne

def drop_overlap(rec: np.ndarray):
    # ch_idx, start_sec, end_sec, ev_type
    rec = rec[:, :4] # sometimes there is a fifth col full of nan
    
    # rec[:, 3] = rec[:, 3] > 3 # 1, 2, 3 -> 0; 4, 5, 6 -> 1
    rec = np.unique(rec[:, 1:], axis=0) # start_sec, end_sec, ev_type
    rec = rec[rec[:, 0].argsort()] # sort by startsec, might be unnecessary
    return np.hstack((-np.ones((rec.shape[0], 1)), rec))

def drop_overlap_2class(rec: np.ndarray):
    rec = drop_overlap(rec)
    rec[:, 3] = rec[:, 3] > 3 # 1, 2, 3 -> 0; 4, 5, 6 -> 1
    rec[:, 3] += 1
    return rec

def drop_overlap_3class(rec: np.ndarray):
    # ch_idx, start_sec, end_sec, ev_type
    rec = rec[:, :4] # sometimes there is a fifth col full of nan
    
    rec[:, 3] = rec[:, 3] > 3 # 1, 2, 3 -> 0; 4, 5, 6 -> 1
    rec = np.unique(rec[:, 1:], axis=0) # start_sec, end_sec, ev_type
    rec = rec[rec[:, 0].argsort()] # sort by startsec, might be unnecessary

    reannotated = []
    for _win_start, _win_end in zip(rec[:, 0].tolist(), rec[:, 1].tolist()):
        reannotated_row = [-1, _win_start, _win_end, -1]
        started = (rec[:, 0] <= _win_end) & (rec[:, 1] >= _win_end) # started before end of window, and end after end of window
        yet2end = (rec[:, 1] >= _win_start) & (rec[:, 0] <= _win_start) # started before start of window, and end after start of window
        #         |=== window ===|
        #                  |--- started ---|
        #    |--- yet2end ---|
    
        overlapped_rec = rec[started | yet2end]
        
        # 1: IED: 1, 2, 3
        # 2: ATF: 4, 5, 6
        # 3: BAD: IED ATF overlap
        unique_overlapped_ev_type = np.unique(overlapped_rec[:, 2])
        if len(unique_overlapped_ev_type) == 1:
            reannotated_row[-1] = unique_overlapped_ev_type.item() + 1 # 0 -> 1, 1 -> 2
        else:
            reannotated_row[-1] = 3

        reannotated.append(reannotated_row)
    
    return np.array(reannotated)

def reannotate(rec: np.ndarray, raw: mne.io.Raw):
    # ch_idx, start_sec, end_sec, ev_type
    rec = rec[:, :4] # sometimes there is a fifth col full of nan
    
    last_full_sec = np.floor(raw.times[-1]).item()

    rec[:, 3] = rec[:, 3] > 3 # 1, 2, 3 -> 0; 4, 5, 6 -> 1
    rec = np.unique(rec[:, 1:], axis=0) # start_sec, end_sec, ev_type
    rec = rec[rec[:, 0].argsort()] # sort by startsec, might be unnecessary

    win_start = np.arange(0, last_full_sec, step=1)
    win_end = win_start + 1

    reannotated = []
    for _win_start, _win_end in zip(win_start.tolist(), win_end.tolist()):
        reannotated_row = [-1, _win_start, _win_end, -1]
        started = (rec[:, 0] <= _win_end) & (rec[:, 1] >= _win_end) # started before end of window, and end after end of window
        yet2end = (rec[:, 1] >= _win_start) & (rec[:, 0] <= _win_start) # started before start of window, and end after start of window
        #         |=== window ===|
        #                  |--- started ---|
        #    |--- yet2end ---|
    
        overlapped_rec = rec[started | yet2end]
        
        # 1: NUL: None
        # 2: IED: 1, 2, 3
        # 3: ATF: 4, 5, 6
        # 4: BAD: IED ATF overlap
        if len(overlapped_rec) == 0:
            reannotated_row[-1] = 1
        else:
            unique_overlapped_ev_type = np.unique(overlapped_rec[:, 2])
            if len(unique_overlapped_ev_type) == 1:
                reannotated_row[-1] = unique_overlapped_ev_type.item() + 2 # 0 -> 2, 1 -> 3
            else:
                reannotated_row[-1] = 4

        reannotated.append(reannotated_row)

    return np.array(reannotated)

def reannotate_2class(rec: np.ndarray, raw: mne.io.Raw):
    # ch_idx, start_sec, end_sec, ev_type
    rec = rec[:, :4] # sometimes there is a fifth col full of nan
    
    last_full_sec = np.floor(raw.times[-1]).item()

    rec[:, 3] = rec[:, 3] > 3 # 1, 2, 3 -> 0; 4, 5, 6 -> 1
    rec = np.unique(rec[:, 1:], axis=0) # start_sec, end_sec, ev_type
    rec = rec[rec[:, 0].argsort()] # sort by startsec, might be unnecessary

    win_start = np.arange(0, last_full_sec, step=1)
    win_end = win_start + 1

    reannotated = []
    for _win_start, _win_end in zip(win_start.tolist(), win_end.tolist()):
        reannotated_row = [-1, _win_start, _win_end, -1]
        started = (rec[:, 0] <= _win_end) & (rec[:, 1] >= _win_end) # started before end of window, and end after end of window
        yet2end = (rec[:, 1] >= _win_start) & (rec[:, 0] <= _win_start) # started before start of window, and end after start of window
        #         |=== window ===|
        #                  |--- started ---|
        #    |--- yet2end ---|
    
        overlapped_rec = rec[started | yet2end]
        
        # 1: NUL: None + 4, 5, 6 + overlap
        # 2: IED: 1, 2, 3
        if len(overlapped_rec) == 0:
            reannotated_row[-1] = 1
        else:
            unique_overlapped_ev_type = np.unique(overlapped_rec[:, 2])
            if len(unique_overlapped_ev_type) == 1:
                reannotated_row[-1] = 1 if unique_overlapped_ev_type.item() == 1 else 2 # 0 -> 2, 1 -> 1
            else:
                reannotated_row[-1] = 1

        reannotated.append(reannotated_row)

    return np.array(reannotated)

def reannotate_2class_onset(rec: np.ndarray, raw: mne.io.Raw):
    # ch_idx, start_sec, end_sec, ev_type
    rec = rec[:, :4] # sometimes there is a fifth col full of nan
    
    last_full_sec = np.floor(raw.times[-1]).item()

    rec[:, 3] = rec[:, 3] > 3 # 1, 2, 3 -> 0; 4, 5, 6 -> 1
    rec = np.unique(rec[:, 1:], axis=0) # start_sec, end_sec, ev_type
    rec = rec[rec[:, 0].argsort()] # sort by startsec, might be unnecessary

    win_start = np.arange(0, last_full_sec, step=1)
    win_end = win_start + 1

    reannotated = []
    for _win_start, _win_end in zip(win_start.tolist(), win_end.tolist()):
        reannotated_row = [-1, _win_start, _win_end, -1]
        started = (rec[:, 0] <= _win_end) & (rec[:, 1] >= _win_end) # started before end of window, and end after end of window
        # yet2end = (rec[:, 1] >= _win_start) & (rec[:, 0] <= _win_start) # started before start of window, and end after start of window
        #         |=== window ===|
        #                  |--- started ---|
        #    |--- yet2end ---|
    
        overlapped_rec = rec[started]
        
        # 1~5: IED: 1, 2, 3, onset at the first 0.2 sec window, second 0.2 sec window ... fifth 0.2 sec window
        # 6: NUL: None + 4, 5, 6 + overlap
        if len(overlapped_rec) == 0:
            reannotated_row[-1] = 6
        else:
            unique_overlapped_ev_type = np.unique(overlapped_rec[:, 2])
            if len(unique_overlapped_ev_type) == 1:
                if unique_overlapped_ev_type.item() == 1:
                    reannotated_row[-1] = 6 
                else:
                    zero_point_2_window_index = int((overlapped_rec[0][0] - _win_start) / 0.2)
                    if zero_point_2_window_index == 5: continue
                    assert zero_point_2_window_index in [0, 1, 2, 3, 4], zero_point_2_window_index
                    reannotated_row[-1] = zero_point_2_window_index + 1
            else:
                reannotated_row[-1] = 6

        reannotated.append(reannotated_row)

    return np.array(reannotated)

import torch
import torch.nn.functional as F
class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        gamma: float = 0,
        mode = "multiclass",
        pos_weight = None,
        # is_binary: bool = False
    ):
        super().__init__()
        self.weight = None if weight is None else torch.nn.Parameter(torch.tensor(weight).float().flatten(), requires_grad=False)
        match reduction:
            case "mean": self.reduce_fn = torch.mean
            case "sum": self.reduce_fn = torch.sum
            case _: raise NotImplementedError()
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.mode = mode
        self.pos_weight = None if pos_weight is None else torch.nn.Buffer(torch.tensor([pos_weight]))
        if mode == "binary":
            assert weight is None
        elif mode == "multiclass":
            assert pos_weight == None
        # self.is_binary = is_binary
        # if self.is_binary: assert label_smoothing == 0
    
    def forward(self, pred, target):
        self_weight = self.weight.to(device=pred.device)
        match self.mode:
            case "multiclass":
                ce_loss = F.cross_entropy(pred, target, weight=self_weight, reduction="none", label_smoothing=self.label_smoothing)
            case "binary":
                _target = target.float() * (1 - self.label_smoothing) + (self.label_smoothing / 2)
                ce_loss = F.binary_cross_entropy_with_logits(pred, _target, pos_weight=self.pos_weight, reduction="none")
        if self.gamma == 0: return self.reduce_fn(ce_loss)

        match self.mode:
            case "multiclass": prob = F.softmax(pred, dim=-1) * F.one_hot(target, num_classes=pred.shape[-1])
            case "binary": prob = F.sigmoid(pred)
        confidence = prob.sum(dim=-1, keepdim=True)
        focal_weight = (1 - confidence) ** self.gamma
        ce_loss = focal_weight * ce_loss

        return self.reduce_fn(ce_loss)
