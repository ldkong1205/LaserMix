import torch
import torch.nn as nn
from torch.nn import functional as F


class FIDNet(nn.Module):

    def __init__(
        self,
        num_cls: int
    ):
        super(FIDNet, self).__init__()
        self.backend=Backbone(if_BN=True, if_remission=True, if_range=True, with_normal=False)
        self.semantic_head=SemanticHead(num_cls=num_cls, input_channel=1024)

    def forward(self, x):
        middle_feature_maps=self.backend(x)  # [bs, 1024, H, W]
        semantic_output=self.semantic_head(middle_feature_maps)  # [bs, cls, H, W]
        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        if_BN: bool = False,
    ):
        super(BasicBlock, self).__init__()

        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()

        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        if_BN: bool = False,
    ):
        super(Bottleneck, self).__init__()
        self.if_BN = if_BN

        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        if self.if_BN:
            self.bn1 = norm_layer(width)
        
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if self.if_BN:
            self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1(width, planes * self.expansion)
        if self.if_BN:
            self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.if_BN:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class SemanticHead(nn.Module):

    def __init__(
        self,
        num_cls: int = 20,
        input_channel: int = 1024,
    ):
        super(SemanticHead,self).__init__()

        self.conv_1=nn.Conv2d(input_channel, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu_1 = nn.LeakyReLU()

        self.conv_2=nn.Conv2d(512, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.LeakyReLU()

        self.semantic_output=nn.Conv2d(128, num_cls, 1)

    def forward(self, input_tensor):
        res=self.conv_1(input_tensor)   # [bs, 512, H, W]
        res=self.bn1(res)
        res=self.relu_1(res)
        
        res=self.conv_2(res)            # [bs, 128, H, W]
        res=self.bn2(res)
        res=self.relu_2(res)
        
        res=self.semantic_output(res)   # [bs, cls, H, W]
        return res


class SemanticBackbone(nn.Module):

    def __init__(
        self,
        block,
        layers,
        if_BN: bool,
        if_remission: bool,
        if_range: bool,
        with_normal: bool,
        norm_layer = None,
        groups: int = 1,
        width_per_group: int = 64,
    ):
        super(SemanticBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN=if_BN
        self.if_remission=if_remission
        self.if_range=if_range
        self.with_normal=with_normal
        self.inplanes = 512
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if not self.if_remission and not self.if_range and not self.with_normal:        
            self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and not self.if_range and not self.with_normal:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and self.if_range and not self.with_normal:
            self.conv1 = nn.Conv2d(6, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and self.if_range and self.with_normal:
            self.conv1 = nn.Conv2d(9, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn_2 = nn.BatchNorm2d(512)
        self.relu_2 = nn.LeakyReLU()
        
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, if_BN=self.if_BN))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):  # [bs, 6, H, W]

        x = self.conv1(x)  # [bs, 64, H, W]
        x = self.bn_0(x)
        x = self.relu_0(x)

        x = self.conv2(x)  # [bs, 128, H, W]
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv3(x)  # [bs, 256, H, W]
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv4(x)  # [bs, 512, H, W]
        x = self.bn_2(x)
        x = self.relu_2(x)

        x_1 = self.layer1(x)     # [bs, 128, H, W]
        x_2 = self.layer2(x_1)   # [bs, 128, H/2, W/2]
        x_3 = self.layer3(x_2)   # [bs, 128, H/4, W/4]
        x_4 = self.layer4(x_3)   # [bs, 128, H/8, W/8]

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)  # [bs, 128, H, W]
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)  # [bs, 128, H, W]
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)  # [bs, 128, H, W]
        
        res=[x, x_1, res_2, res_3, res_4]

        return torch.cat(res, dim=1)

    def forward(self, x):
        return self._forward_impl(x)


def _backbone(arch, block, layers, if_BN, if_remission, if_range, with_normal):
    model = SemanticBackbone(block, layers, if_BN, if_remission, if_range, with_normal)
    return model


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride,
        bias=False,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation,
    )


def Backbone(if_BN, if_remission, if_range, with_normal):
    """ResNet-34 model from "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>"""
    return _backbone(
        'resnet34', BasicBlock, [3, 4, 6, 3],
        if_BN, if_remission, if_range, with_normal,
    )

