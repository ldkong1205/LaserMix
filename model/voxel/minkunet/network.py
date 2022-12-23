from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as TSF

from torchsparse import PointTensor, SparseTensor
from torchsparse.utils import make_ntuple
from torchsparse.nn.utils import fapply



def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1,
    )
    hash_pc = TSF.sphash(torch.floor(new_float_coord).int())
    hash_sparse = torch.unique(hash_pc)
    idx_query = TSF.sphashquery(hash_pc, hash_sparse)
    counts = TSF.spcount(idx_query.int(), len(hash_sparse))

    inserted_coords = TSF.spvoxelize(torch.floor(new_float_coord), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_fea = TSF.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_fea, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


def get_kernel_offsets(
    size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    device: str = 'cpu',
) -> torch.Tensor:
    size = make_ntuple(size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    offsets = [(np.arange(-size[k] // 2 + 1, size[k] // 2 + 1) * stride[k]
                * dilation[k]) for k in range(3)]

    if np.prod(size) % 2 == 1:
        offsets = [[x, y, z] for z in offsets[2] for y in offsets[1]
                   for x in offsets[0]]
    else:
        offsets = [[x, y, z] for x in offsets[0] for y in offsets[1]
                   for z in offsets[2]]

    offsets = torch.tensor(offsets, dtype=torch.int, device=device)
    return offsets


def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None \
        or z.idx_query.get(x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        hash_old = TSF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off,
        )
        hash_pc = TSF.sphash(x.C.to(z.F.device))
        idx_query = TSF.sphashquery(hash_old, hash_pc)
        weights = TSF.calc_ti_weights(
            z.C, idx_query, scale=x.s[0]
        ).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()

        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = TSF.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(
            new_feat, z.C,
            idx_query=z.idx_query,
            weights=z.weights,
        )
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = TSF.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(
            new_feat, z.C,
            idx_query=z.idx_query,
            weights=z.weights,
        )
        new_tensor.additional_features = z.additional_features

    return new_tensor


class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BasicConvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNet(nn.Module):

    def __init__(
        self,
        num_cls: int,
        num_layer: list,
        cr: float,
        plane: list,
        in_fea_dim: int,
        ignore_label: int = 0,
        if_dist: bool = False,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.in_fea_dim = in_fea_dim
        self.ignore_label = ignore_label

        self.num_layer = num_layer
        self.block = {
            'ResBlock': ResidualBlock,
            'Bottleneck': Bottleneck,
        }['ResBlock']

        cr = cr
        cs = plane
        cs = [int(cr * x) for x in cs]

        self.pres = 0.05
        self.vres = 0.05

        self.stem = nn.Sequential(
            spnn.Conv3d(
                self.in_fea_dim, cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(
                cs[0], cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.in_channels = cs[0]

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[1], self.num_layer[0]),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[2], self.num_layer[1]),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[3], self.num_layer[2]),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[4], self.num_layer[3]),
        )

        self.up1 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[5],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(
            nn.Sequential(*self._make_layer(self.block, cs[5], self.num_layer[4]))
        )
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[6],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(
            nn.Sequential(*self._make_layer(self.block, cs[6], self.num_layer[5]))
        )
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[7],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(
            nn.Sequential(*self._make_layer(self.block, cs[7], self.num_layer[6]))
        )
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[8],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(
            nn.Sequential(*self._make_layer(self.block, cs[8], self.num_layer[7]))
        )
        self.up4 = nn.ModuleList(self.up4)

        self.classifier = nn.Sequential(
            nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_cls)
        )

        self.weight_initialization()

        dropout_p = 0.0
        self.dropout = nn.Dropout(dropout_p, True)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return layers
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        x = batch['point_fea']
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)

        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        z1 = voxel_to_point(x4, z0)

        x4.F = self.dropout(x4.F)
        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)

        y2.F = self.dropout(y2.F)
        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)

        logits = self.classifier(torch.cat([z1.F, z2.F, z3.F], dim=1))

        label = batch['point_label'].F.long().cuda(non_blocking=True)

        return logits, label

