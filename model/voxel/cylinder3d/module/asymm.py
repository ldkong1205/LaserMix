import numpy as np
import torch
from torch import nn

import spconv.pytorch as spconv


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, padding=1,
        bias=False, indice_key=indice_key,
    )

def conv1x3(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1),
        bias=False, indice_key=indice_key,
    )

def conv3x1(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=(3, 1, 3), stride=stride, padding=(1, 0, 1),
        bias=False, indice_key=indice_key,
    )

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1,
        bias=False, indice_key=indice_key,
    )

def conv1x1x3(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=(1, 1, 3), stride=stride, padding=(0, 0, 1),
        bias=False, indice_key=indice_key,
    )

def conv1x3x1(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 1), stride=stride, padding=(0, 1, 0),
        bias=False, indice_key=indice_key,
    )

def conv3x1x1(
    in_planes: int, out_planes: int, stride: int = 1, indice_key = None,
):
    return spconv.SubMConv3d(
        in_planes, out_planes,
        kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0),
        bias=False, indice_key=indice_key,
    )


class ResContextBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        indice_key = None
    ):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef"
        )
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + "1"
        )
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef"
        )
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key + "2"
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)  # [480, 360, 32]
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        pooling: bool = True,
        drop_out: bool = True,
        height_pooling: bool = False,
        indice_key = None,
    ):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef"
        )
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key + "1"
        )
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef"
        )
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + "a"
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(
                    out_filters, out_filters,
                    kernel_size=3, stride=2, padding=1,
                    indice_key=indice_key, bias=False,
                )
            else:
                self.pool = spconv.SparseConv3d(
                    out_filters, out_filters,
                    kernel_size=3, stride=(2, 2, 1), padding=1,
                    indice_key=indice_key, bias=False,
                )
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

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
        indice_key = None,
        up_key = None,
    ):
        super(UpBlock, self).__init__()
        self.trans_dilao = conv3x3(
            in_filters, out_filters,
            indice_key=indice_key + "new_up",
        )
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + '1',
        )
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(
            out_filters, out_filters,
            indice_key=indice_key + '7',
        )
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.up_subm = spconv.SparseInverseConv3d(
            out_filters, out_filters,
            kernel_size=3,
            indice_key=up_key, bias=False,
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        upA = self.up_subm(upA)
        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        indice_key = None,
    ):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class asymm_3d_spconv(nn.Module):
    
    def __init__(
        self,
        nclasses: int,
        output_shape: list = [240, 180, 20],
        num_input_features: int = 32,
        n_height: int = 32,
        init_size: int = 16,
    ):
        super(asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height  # 32
        self.strict = False
        self.sparse_shape = np.array(output_shape)  # [240, 180, 20]

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        
        self.resBlock2 = ResBlock(init_size,   2*init_size,  height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2*init_size, 4*init_size,  height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4*init_size, 8*init_size,  pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8*init_size, 16*init_size, pooling=True, height_pooling=False, indice_key="down5")

        self.upBlock0 = UpBlock(16*init_size, 16*init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16*init_size, 8*init_size,  indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8*init_size,  4*init_size,  indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4*init_size,  2*init_size,  indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2*init_size, 2*init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, voxel_features, coors, batch_size):  # voxel_features: [uniq, 16], coors: [uniq, 4], batch_size: bs
        coors = coors.int()  # [uniq, 4]

        ret = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )  # [uniq, 16], [uniq, 4], (3,), bs -> [240, 180, 32]
        ret = self.downCntx(ret)  # [480, 360, 32]
        
        down1c, down1b = self.resBlock2(ret)     # [240, 180, 16], [480, 360, 32]
        down2c, down2b = self.resBlock3(down1c)  # [120, 90,  8 ], [240, 180, 16]
        down3c, down3b = self.resBlock4(down2c)  # [60,  45,  8 ], [120, 90,  8 ]
        down4c, down4b = self.resBlock5(down3c)  # [30,  23,  8 ], [60,  45,  8 ]

        up4e = self.upBlock0(down4c, down4b)  # [60,  45,  8 ]
        up3e = self.upBlock1(up4e,   down3b)  # [120, 90,  8 ]
        up2e = self.upBlock2(up3e,   down2b)  # [240, 180, 16]
        up1e = self.upBlock3(up2e,   down1b)  # [480, 360, 32]

        up0e = self.ReconNet(up1e)  # [480, 360, 32]

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1)) # [uniq, 64]

        logits = self.logits(up0e)  # [480, 360, 32]
        y = logits.dense()  # [bs, cls, 480, 360, 32]

        return y
