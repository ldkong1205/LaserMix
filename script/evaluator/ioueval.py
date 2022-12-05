import numpy as np
import torch


class iouEval:
    def __init__(self, n_classes:int, ignore:int = 0):
        self.n_classes = n_classes
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor([n for n in range(self.n_classes) if n not in self.ignore]).long()
        print("[IOU EVAL] IGNORED CLASS: ", self.ignore)
        print("[IOU EVAL] INCLUDE CLASS: ", self.include)
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes)).long().cuda()
        self.ones = None
        self.last_scan_size = None

    def addBatch(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().cuda()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().cuda()

        x_row = x.reshape(-1)
        y_row = y.reshape(-1)

        idxs = torch.stack([x_row, y_row], dim=0)

        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1])).long().cuda()
            self.last_scan_size = idxs.shape[-1]

        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

    def getStats(self):
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean
        