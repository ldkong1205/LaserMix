import torch
from torch import nn
import torch.nn.functional

import torch_scatter
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional
from torchsparse import PointTensor, SparseTensor


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False,
    )

def conv1x3(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 3), stride=stride, bias=False,
    )

def conv3x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(3, 1, 3), stride=stride, bias=False,
    )

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, bias=False,
    )

def conv1x1x3(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 1, 3), stride=stride, bias=False,
    )

def conv1x3x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 1), stride=stride, bias=False,
    )

def conv3x1x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(3, 1, 1), stride=stride, bias=False,
    )

def torch_unique(x):
    unique, inverse = torch.unique(x, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, inds, inverse

def voxelize(z, init_res=None, after_res=None, voxel_type='max'):
    pc_hash = torchsparse.nn.functional.sphash(z.C.int())
    sparse_hash, inds, idx_query = torch_unique(pc_hash)
    counts = torchsparse.nn.functional.spcount(idx_query.int(), len(sparse_hash))
    inserted_coords = z.C[inds].int()

    if voxel_type == 'max':
        inserted_feat = torch_scatter.scatter_max(z.F, idx_query, dim=0)[0]
    elif voxel_type == 'mean':
        inserted_feat = torch_scatter.scatter_mean(z.F, idx_query, dim=0)
    else:
        raise NotImplementedError
    
    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts

    return new_tensor

class ResContextBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple = (3, 3, 3),
        stride: int = 1,
        indice_key: str = None,
        if_dist: bool = False,
    ):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.act1(shortcut.F)
        shortcut.F = self.bn0(shortcut.F)

        shortcut = self.conv1_2(shortcut)
        shortcut.F = self.act1_2(shortcut.F)
        shortcut.F = self.bn0_2(shortcut.F)

        resA = self.conv2(x)
        resA.F = self.act2(resA.F)
        resA.F = self.bn1(resA.F)

        resA = self.conv3(resA)
        resA.F = self.act3(resA.F)
        resA.F = self.bn2(resA.F)
        resA.F = resA.F + shortcut.F

        return resA


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        dropout_rate: float,
        kernel_size: tuple = (3, 3, 3),
        stride: int = 1,
        pooling: bool = True,
        drop_out: bool = True,
        height_pooling: bool = False,
        indice_key: str = None,
        if_dist: bool = False,
    ):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spnn.Conv3d(
                    out_filters, out_filters,
                    kernel_size=3, stride=2, bias=False,
                )
            else:
                self.pool = spnn.Conv3d(
                    out_filters, out_filters,
                    kernel_size=3, stride=(2, 2, 1), bias=False,
                )
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.act1(shortcut.F)
        shortcut.F = self.bn0(shortcut.F)

        shortcut = self.conv1_2(shortcut)
        shortcut.F = self.act1_2(shortcut.F)
        shortcut.F = self.bn0_2(shortcut.F)

        resA = self.conv2(x)
        resA.F = self.act2(resA.F)
        resA.F = self.bn1(resA.F)

        resA = self.conv3(resA)
        resA.F = self.act3(resA.F)
        resA.F = self.bn2(resA.F)

        resA.F = resA.F + shortcut.F

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple = (3, 3, 3),
        indice_key: str = None,
        up_key: str = None,
        height_pooling: bool = False,
        if_dist: bool = False,
    ):
        super(UpBlock, self).__init__()
        self.trans_dilao = conv3x3(
            in_filters, out_filters,
            indice_key=indice_key + "new_up",
        )
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        if height_pooling:
            self.up_subm = spnn.Conv3d(
                out_filters, out_filters,
                kernel_size=3, stride=2, bias=False, transposed=True,
            )
        else:
            self.up_subm = spnn.Conv3d(
                out_filters, out_filters,
                kernel_size=3, stride=(2, 2, 1), bias=False, transposed=True,
            )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA.F = self.trans_act(upA.F)
        upA.F = self.trans_bn(upA.F)

        upA = self.up_subm(upA)
        upA.F = upA.F + skip.F

        upE = self.conv1(upA)
        upE.F = self.act1(upE.F)
        upE.F = self.bn1(upE.F)

        upE = self.conv2(upE)
        upE.F = self.act2(upE.F)
        upE.F = self.bn2(upE.F)

        upE = self.conv3(upE)
        upE.F = self.act3(upE.F)
        upE.F = self.bn3(upE.F)

        return upE


class ReconBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple = (3, 3, 3),
        stride: int = 1,
        indice_key: str = None,
        if_dist: bool = False,
    ):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0_2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0_3 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.bn0(shortcut.F)
        shortcut.F = self.act1(shortcut.F)

        shortcut2 = self.conv1_2(x)
        shortcut2.F = self.bn0_2(shortcut2.F)
        shortcut2.F = self.act1_2(shortcut2.F)

        shortcut3 = self.conv1_3(x)
        shortcut3.F = self.bn0_3(shortcut3.F)
        shortcut3.F = self.act1_3(shortcut3.F)
        shortcut.F = shortcut.F + shortcut2.F + shortcut3.F

        shortcut.F = shortcut.F * x.F

        return shortcut


class Cylinder3D(nn.Module):

    def __init__(
        self,
        num_cls: int,
        in_fea_dim: int,
        init_size: int = 16,
        ignore_label: int = 0,
        if_point_refinement: bool = False,
        if_dist: bool = False,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.in_fea_dim = in_fea_dim
        self.init_size = init_size
        self.ignore_label = ignore_label
        self.if_point_refinement = if_point_refinement
        
        self.PPmodel = nn.Sequential(
            nn.SyncBatchNorm(self.in_fea_dim) if if_dist else nn.BatchNorm1d(self.in_fea_dim),
            nn.Linear(self.in_fea_dim, 64),
            nn.SyncBatchNorm(64) if if_dist else nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.SyncBatchNorm(128) if if_dist else nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.SyncBatchNorm(256) if if_dist else nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.fea_compression = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU()
        )

        self.downCntx = ResContextBlock(
            16, self.init_size,
            indice_key="pre", if_dist=if_dist,
        )

        self.resBlock2 = ResBlock(
            self.init_size, 2 * self.init_size, 0.2,
            height_pooling=True, indice_key="down2", if_dist=if_dist,
        )
        self.resBlock3 = ResBlock(
            2 * self.init_size, 4 * self.init_size, 0.2,
            height_pooling=True, indice_key="down3", if_dist=if_dist,
        )
        self.resBlock4 = ResBlock(
            4 * self.init_size, 8 * self.init_size, 0.2,
            pooling=True, height_pooling=False, indice_key="down4", if_dist=if_dist,
        )
        self.resBlock5 = ResBlock(
            8 * self.init_size, 16 * self.init_size, 0.2,
            pooling=True, height_pooling=False, indice_key="down5", if_dist=if_dist,
        )

        self.upBlock0 = UpBlock(
            16 * self.init_size, 16 * self.init_size,
            indice_key="up0", up_key="down5", height_pooling=False, if_dist=if_dist,
        )
        self.upBlock1 = UpBlock(
            16 * self.init_size, 8 * self.init_size,
            indice_key="up1", up_key="down4", height_pooling=False, if_dist=if_dist,
        )
        self.upBlock2 = UpBlock(
            8 * self.init_size, 4 * self.init_size,
            indice_key="up2", up_key="down3", height_pooling=True, if_dist=if_dist,
        )
        self.upBlock3 = UpBlock(
            4 * self.init_size, 2 * self.init_size,
            indice_key="up3", up_key="down2", height_pooling=True, if_dist=if_dist,
        )

        self.ReconNet = ReconBlock(
            2 * self.init_size, 2 * self.init_size,
            indice_key="recon", if_dist=if_dist,
        )

        self.logits = spnn.Conv3d(
            4 * self.init_size, self.num_cls,
            kernel_size=3, stride=1, bias=True,
        )

        if self.if_point_refinement:
            self.change_dim = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                nn.SyncBatchNorm(256) if if_dist else nn.BatchNorm1d(256),
                nn.LeakyReLU()
            )
            self.point_logits = torch.nn.Linear(256, self.num_cls)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        point_fea = batch['point_fea']  # [bs*N, 9]
        point_fea = self.PPmodel(point_fea)  # [bs*N, 256]

        z = PointTensor(point_fea, batch['point_coord'].float())

        ret = voxelize(z)
        ret.F = self.fea_compression(ret.F)  # [uniq, 16]
        ret = self.downCntx(ret)

        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)
        up0e.F = torch.cat((up0e.F, up1e.F), 1)  # [uniq, 64]

        logits = self.logits(up0e).F  # [uniq, cls]

        if self.if_point_refinement:
            hash_point = torchsparse.nn.functional.sphash(batch['point_coord'].to(logits).int())
            hash_voxel = torchsparse.nn.functional.sphash(batch['voxel_coord'].to(logits).int())
            idx_query = torchsparse.nn.functional.sphashquery(hash_point, hash_voxel)

            point_fea_from_voxel = up0e.F[idx_query]
            point_fea_from_voxel = self.change_dim(point_fea_from_voxel)
            point_fea_cat = point_fea + point_fea_from_voxel
            point_logits = self.point_logits(point_fea_cat)
            point_label = batch['point_label']
            
            return point_logits, point_label

        else:
            hash_voxel = torchsparse.nn.functional.sphash(batch['voxel_coord'].to(up0e.C))
            hash_logits = torchsparse.nn.functional.sphash(up0e.C)
            idx_query = torchsparse.nn.functional.sphashquery(hash_logits, hash_voxel)
            
            voxel_label = batch['voxel_label']
            voxel_label = voxel_label[idx_query]

            return logits, voxel_label
        
