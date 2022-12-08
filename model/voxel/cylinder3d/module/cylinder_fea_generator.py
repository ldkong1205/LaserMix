import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(
        self,
        grid_size: list,
        fea_dim: int = 3,
        out_pt_fea_dim: int = 64,
        max_pt_per_encode: int = 64,
        fea_compre = None,
    ):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(
            kernel_size, stride=1, padding=(kernel_size - 1) // 2,
            dilation=1
        )
        self.pool_dim = out_pt_fea_dim

        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):
        cur_dev = pt_fea[0].get_device()

        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))  # [bs, [N, 4]]

        cat_pt_fea = torch.cat(pt_fea, dim=0)  # [N + N, 9]
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)  # [N + N, 4]
        pt_num = cat_pt_ind.shape[0]  # N + N

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)  # [N + N]
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]  # [N + N, 9]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]  # [N + N, 4]

        # unique xy grid index
        unq, unq_inv, _ = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)  # [uniq, 4], [N + N], uniq
        unq = unq.type(torch.int64)  # [uniq, 4]

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)  # [N + N, 9] -> [N + N， 256]
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]  # [uniq, 256]

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)  # [uniq, 16]
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data  # [uniq, 4], [uniq, 16]
