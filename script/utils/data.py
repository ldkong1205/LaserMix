import numpy as np
import torch


def collate_batch(data):
    grid_ind_stack = [d[0] for d in data]  # [N, 3]
    label2stack = np.stack([d[1] for d in data]).astype(np.int)  # [bs, 240, 180, 32]
    p_fea = [d[2] for d in data]  # [N, 9]
    p_label = [d[3] for d in data]  # [N, 1]
    return grid_ind_stack, torch.from_numpy(label2stack), p_fea, p_label,
