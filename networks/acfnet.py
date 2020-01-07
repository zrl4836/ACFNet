import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
import sys, os
from utils.pyt_utils import load_model

from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)

        self.eca = eca_layer(planes * 4, k_size=3)

        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class acf_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels, out_channels):
        super(acf_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, feat_ffm, coarse_x):
        """
            inputs :
                feat_ffm : input feature maps( B X C X H X W), C is channel
                coarse_x : input feature maps( B X N X H X W), N is class
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, N, height, width = coarse_x.size()

        # CCB: Class Center Block start...
        # 1x1conv -> F' 
        feat_ffm = self.conv1(feat_ffm)
        b, C, h, w = feat_ffm.size()

        # P_coarse reshape ->(B, N, W*H)
        proj_query = coarse_x.view(m_batchsize, N, -1)

        # F' reshape and transpose -> (B, W*H, C')
        proj_key = feat_ffm.view(b, C, -1).permute(0, 2, 1)

        # multiply & normalize ->(B, N, C')
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # CCB: Class Center Block end...

        # CAB: Class Attention Block start...
        # transpose ->(B, C', N)
        attention = attention.permute(0, 2, 1)

        # (B, N, W*H)
        proj_value = coarse_x.view(m_batchsize, N, -1)

        # # multiply (B, C', N)(B, N, W*H)-->(B, C, W*H)
        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        # 1x1conv
        out = self.conv2(out)
        # CAB: Class Attention Block end...

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ACFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACFModule, self).__init__()
        
        #self.conva = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        
        self.acf = acf_Module(in_channels, out_channels)
<<<<<<< HEAD
        
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1, bias=False),
        #     InPlaceABNSync(256),
        #     nn.Dropout2d(0.1,False),
        #     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )

    def forward(self, x, coarse_x):
        class_output = self.acf(x, coarse_x)
        #feat_cat = torch.cat([class_output, output],dim=1)
        return class_output


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), mode='bilinear', align_corners=True)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(ASPP_Module, self).__init__()
        # In our re-implementation of ASPP module,
        # we follow the original paper but change the output channel
        # from 256 to 512 in all of four branches.
        out_channels = in_channels // 4
        
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer)


    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return y


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
=======
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, coarse_x):
        class_output = self.acf(x, coarse_x)
        output = self.conva(x)
        # CONCAT or SUM
        feat_sum = class_output * self.gamma + output
        output = self.bottleneck(feat_sum)
        return output
>>>>>>> 2fd812363efe5e1c2ed6b42afbb2e08c650c8bfa


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        
        self.aspp = ASPP_Module(in_channels=2048, atrous_rates=(12, 24, 36), norm_layer=BatchNorm2d)
        
        self.auxlayer = FCNHead(1024, num_classes, BatchNorm2d)
        
        self.dsn = nn.Sequential(
            nn.Conv2d(2560, 512, 1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.acfhead = ACFModule(2560, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(3072, 1024, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(1024),
            nn.Dropout2d(0.1,False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.criterion = criterion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        feat_aspp = self.aspp(feat32)

        auxout = self.auxlayer(feat16)

        coarse_x = self.dsn(feat_aspp)
        
        acf_out = self.acfhead(feat_aspp, coarse_x)

        feat_cat = torch.cat([acf_out, feat_aspp],dim=1)
        
        pre_out = self.bottleneck(feat_cat)
        
        outs = [pre_out, coarse_x, auxout]
 
        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs


def Seg_Model(num_classes, criterion=None, pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, criterion)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)

    return model
