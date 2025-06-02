import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from torch.nn import Softmax, Dropout
from model.pvtv2 import pvt_v2_b2

from typing import List, Callable
from torch import Tensor

# out = channel_shuffle(out, 2)
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicConv2dRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# SpatialSelfAtt is CPIM
class SpatialSelfAtt(nn.Module):
    def __init__(self, channel=128):
        super(SpatialSelfAtt, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_cur1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_cur2 = nn.Conv2d(channel, channel, kernel_size=1)

        self.gamma_cur1 = nn.Parameter(torch.ones(1))
        self.gamma_cur2 = nn.Parameter(torch.ones(1))
        # following DANet
        self.conv_cur1 = nn.Sequential(BasicConv2dRelu(channel, channel, 3, padding=1),
                                       nn.Dropout2d(0.1, False),
                                       BasicConv2dRelu(channel, channel, 1)
                                       )
        self.conv_cur2 = nn.Sequential(BasicConv2dRelu(channel, channel, 3, padding=1),
                                      nn.Dropout2d(0.1, False),
                                      BasicConv2dRelu(channel, channel, 1)
                                      )

    def forward(self, x_1_up, x_2): # x_1: Q, x_cur: K
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: C X H x W
        """
        x_1 = self.downsample(x_1_up)
        proj_query = self.query_conv(x_1) # B X C X H2 x W2
        proj_key = self.key_conv(x_2) # B X C X H1 x W1
        proj_query_t = torch.transpose(proj_query,2,3).contiguous()  # B X C X W2 x H2 (W=H)
        energy = torch.matmul(proj_query_t, proj_key) # C X (W2 x H2) x (H1 X W1) = C X W2 x W1

        attention1 = F.softmax(torch.transpose(energy, 2, 3), dim=2)  # C X W1 x W2
        proj_value_cur1 = self.value_conv_cur1(x_1)  # C X H1 x W1
        out_cur1 = self.upsample2(torch.matmul(proj_value_cur1, attention1).contiguous())  # C X H1 x W1 X (W1 x W2) = C X H1 X W1
        out_cur1 = self.conv_cur1(self.gamma_cur1 * out_cur1 + x_1_up)

        attention2 = F.softmax(energy.clone(), dim=2) # C X W2 x W1
        proj_value_cur2 = self.value_conv_cur2(x_2)  # C X H1 x W1
        out_cur2 = torch.matmul(proj_value_cur2, attention2).contiguous()  # C X H1 x W1 X (W2 x W1) = C X H1 X W1
        out_cur2 = self.conv_cur2(self.gamma_cur2 * out_cur2 + x_2)

        return out_cur1,out_cur2

# CA_GMP_GAP is CCAU
class CA_GMP_GAP(nn.Module):
    def __init__(self, channel=64, reduction=4):
        super(CA_GMP_GAP, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x3_1, x4_1): # each x_x has 8 channels
        max_result_3 = self.maxpool(x3_1)
        avg_result_3 = self.avgpool(x3_1)
        max_result_4 = self.maxpool(x4_1)
        avg_result_4 = self.avgpool(x4_1)

        # 32 channels
        max_result = channel_shuffle(torch.cat([max_result_3, max_result_4], dim=1),2)
        avg_result = channel_shuffle(torch.cat([avg_result_3, avg_result_4], dim=1),2)

        max_out = self.ca(max_result)
        avg_out = self.ca(avg_result)
        output = self.sigmoid(max_out + avg_out)

        # 8 channels
        x3_1_cam, x4_1_cam = torch.split(output, 32, dim = 1)

        x3_1_ca = x3_1.mul(x3_1_cam)
        x4_1_ca = x4_1.mul(x4_1_cam)

        return x3_1_ca, x4_1_ca

# CCAM is CCIM
class CCAM(nn.Module):
    def __init__(self, channel=128):
        super(CCAM, self).__init__()
        self.CA_GMP_GAP_1 = CA_GMP_GAP()
        self.CA_GMP_GAP_2 = CA_GMP_GAP()
        self.CA_GMP_GAP_3 = CA_GMP_GAP()
        self.CA_GMP_GAP_4 = CA_GMP_GAP()

        # self.fu3 = BasicConv2dRelu(32, 32, 3, padding=1)
        # self.fu4 = BasicConv2dRelu(32, 32, 3, padding=1)
        # self.sa3 = SpatialAttention()
        # self.sa4 = SpatialAttention()

    def forward(self, x3, x4): # each x has 32 channels, split on the channel dim
        x3_1, x3_2, x3_3, x3_4 = torch.split(x3, 32, dim = 1) # each x_x has 8 channels
        x4_1, x4_2, x4_3, x4_4 = torch.split(x4,32, dim = 1)

        x3_1_ca, x4_1_ca = self.CA_GMP_GAP_1(x3_1, x4_1)
        x3_2_ca, x4_2_ca = self.CA_GMP_GAP_2(x3_2, x4_2)
        x3_3_ca, x4_3_ca = self.CA_GMP_GAP_3(x3_3, x4_3)
        x3_4_ca, x4_4_ca = self.CA_GMP_GAP_4(x3_4, x4_4)

        x3_ca = torch.cat([x3_1_ca, x3_2_ca, x3_3_ca, x3_4_ca], dim=1)
        x4_ca = torch.cat([x4_1_ca, x4_2_ca, x4_3_ca, x4_4_ca], dim=1)

        return x3_ca, x4_ca

# SA_GMP_GAP is CSAU
class SA_GMP_GAP(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_GMP_GAP, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1_1, x2_1): # each x_x has 8 channels
        x1_x2 = torch.cat([x1_1, x2_1], dim=2)
        avg_out = torch.mean(x1_x2, dim=1, keepdim=True)
        max_out, _ = torch.max(x1_x2, dim=1, keepdim=True)
        output = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

        B, C, H, W = output.size()
        # 2 channels
        x1_1_sam, x2_1_sam = torch.split(output, H//2, dim = 2)
        x1_1_sa = x1_1.mul(x1_1_sam)
        x2_1_sa = x2_1.mul(x2_1_sam)

        return x1_1_sa, x2_1_sa

# CSAM is CSIM
class CSAM(nn.Module):
    def __init__(self, channel=128):
        super(CSAM, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.SA_GMP_GAP_1 = SA_GMP_GAP()
        self.SA_GMP_GAP_2 = SA_GMP_GAP()

    def forward(self, x1, x2): # each x has 32 channels, split on the spatial dim
        B1, C1, H1, W1 = x1.size()
        x1_1, x1_2 = torch.split(x1, H1//2, dim = 2) # each x_x has 16 channels
        x2_up = self.upsample2(x2)
        B2, C2, H2, W2 = x2_up.size()
        x2_1, x2_2 = torch.split(x2_up, H2//2, dim = 2)

        x1_1_sa, x2_1_sa = self.SA_GMP_GAP_1(x1_1, x2_1)
        x1_2_sa, x2_2_sa = self.SA_GMP_GAP_2(x1_2, x2_2)

        x1_csam = torch.cat([x1_1_sa, x1_2_sa], dim=2)
        x2_csam = self.downsample(torch.cat([x2_1_sa, x2_2_sa], dim=2))

        return x1_csam, x2_csam

# Dynamic Conv
class DynamicConv(nn.Module):
    def __init__(self, channel=128):
        super(DynamicConv, self).__init__()
        self.pointconv = BasicConv2dRelu(channel, channel, 1)

    def forward(self, x, k):  # x:32*11*11 k:32*3*3
        B, C, H, W = k.size()
        x_B, x_C, x_H, x_W = x.size()  # 8*32*11*11

        x_new = x.clone()
        # k = k.view(C, 1, H, W)
        for i in range(0, B): #由1改为了0,groups由C改为了1
            kernel = k[i, :, :, :]
            kernel = kernel.view(C, 1, H, W)
            # DDconv
            x_r1 = F.conv2d(x[i, :, :, :].view(1, C, x_H, x_W), kernel, stride=1, padding=H//2, dilation=1, groups=C)
            # print(x_r1.size())
            x_new[i, :, :, :] = x_r1
        # print(x_new.size())
        x_all = self.pointconv(x_new)

        return x_all

# Semantic enhancement unit
class SemEU(nn.Module):
    def __init__(self, channel=128, kernel=3):
        super(SemEU, self).__init__()

        # self.SA = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(kernel)
        self.DynamicConv = DynamicConv(channel)
        # self.fuse = BasicConv2dRelu(channel, channel, 3, 1, 1)

    def forward(self, sem, x): # sem, x1_CCAM
        # fea_size = x.size()[2:]
        # sem_up = F.interpolate(sem, size=fea_size, mode="bilinear", align_corners=True)

        # x_sa= x.mul(self.SA(torch.cat([sem_up, x], dim=1)))+x

        # 可以不要这里的动态卷积看下哪个效果好
        sem_kernel = self.avgpool(sem)  # 32*5*5
        SemDyConv= self.DynamicConv(x, sem_kernel)
        out = SemDyConv + x

        return out

class DecoderFusion(nn.Module):
    def __init__(self, channel=128):
        super(DecoderFusion, self).__init__()

        # self.SA = SpatialAttention()
        self.prod_fuse = BasicConv2dRelu(channel, channel, 3, 1, 1)
        self.diff_fuse = BasicConv2dRelu(channel, channel, 3, 1, 1)

    def forward(self, x3, x4): # sem, x1_CCAM
        x_prod = self.prod_fuse(x3*x4)
        x_diff = self.diff_fuse(torch.abs(x3-x4))
        out = x_prod + x_diff

        return out

#SalReasoner is Defect Reasoner
class SalReasoner(nn.Module):
    def __init__(self, channel):
        super(SalReasoner, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.HL4 = nn.Sequential(
            BasicConv2dRelu(channel, channel, 3, 1, 1),
            BasicConv2dRelu(channel, channel, 3, 1, 1),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.HL3 = nn.Sequential(
            BasicConv2dRelu(2*channel, channel, 3, 1, 1),
            BasicConv2dRelu(channel, channel, 3, 1, 1),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.HL2 = nn.Sequential(
            BasicConv2dRelu(2*channel, channel, 3, 1, 1),
            BasicConv2dRelu(channel, channel, 3, 1, 1),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.HL1 = nn.Sequential(
            BasicConv2dRelu(2*channel, channel, 3, 1, 1),
            BasicConv2dRelu(channel, channel, 3, 1, 1),
        )
        self.S1 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4): # x1: 32x384x88, x2: 32x192x44, x3: 32x96x22, x4: 32x48x11, x5: 32x24x11

        x4_HL4 = self.HL4(x4)  # 22
        # x34 = self.DF34(x3,x4_HL4)
        s4 = self.S4(x4_HL4)
        x3_HL3 = self.HL3(torch.cat((x4_HL4, x3), 1))  # 44
        # x23 = self.DF23(x2,x3_HL3)
        s3 = self.S3(x3_HL3)
        x2_HL2 = self.HL2(torch.cat((x3_HL3, x2), 1))  # 88
        # x12 = self.DF23(x1,x2_HL2)
        s2 = self.S2(x2_HL2)
        x1_HL1 = self.HL1(torch.cat((x2_HL2, x1), 1))  # 88
        s1 = self.S2(x1_HL1)

        return s1, s2, s3, s4


class OCINet(nn.Module):
    def __init__(self, channel=128):
        super(OCINet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2dRelu(64, channel, 3, 1, 1)  # 64x384x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2dRelu(128, channel, 3, 1, 1) # 128x192x44->32x44x44
        self.ChannelNormalization_3 = BasicConv2dRelu(320, channel, 3, 1, 1) # 320x96x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2dRelu(512, channel, 3, 1, 1) # 512x48x11->32x11x11

        # SpatialSelfAtt is CPIM
        self.SpatialSelfAtt34 = SpatialSelfAtt()
        self.SpatialSelfAtt23 = SpatialSelfAtt()
        self.SpatialSelfAtt12 = SpatialSelfAtt()
        self.SSA_f3 = BasicConv2dRelu(channel, channel, 3, 1, 1)
        self.SSA_f2 = BasicConv2dRelu(channel, channel, 3, 1, 1)

        # CCAM is CCIM
        self.CollaborativeChannelAttention34 = CCAM(channel)
        self.CollaborativeChannelAttention23 = CCAM(channel)
        self.CollaborativeChannelAttention12 = CCAM(channel)
        self.CCAM_f3 = BasicConv2dRelu(channel, channel, 3, 1, 1)
        self.CCAM_f2 = BasicConv2dRelu(channel, channel, 3, 1, 1)

        # CSAM is CSIM
        self.CollaborativeSpatialAttention12 = CSAM()
        self.CollaborativeSpatialAttention23 = CSAM()
        self.CollaborativeSpatialAttention34 = CSAM()
        self.CSAM_f3 = BasicConv2dRelu(channel, channel, 3, 1, 1)
        self.CSAM_f2 = BasicConv2dRelu(channel, channel, 3, 1, 1)

        # SalReasoner is Defect Reasoner
        self.SalReasoner = SalReasoner(channel)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # 64x88x88
        x2 = pvt[1]  # 128x44x44
        x3 = pvt[2]  # 320x22x22
        x4 = pvt[3]  # 512x11x11

        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2) # 32x22x22
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4) # 32x11x11

        # CCIM
        x3_CCAM_1, x4_CCAM = self.CollaborativeChannelAttention34(x3_nor, x4_nor)
        x2_CCAM_1, x3_CCAM_2 = self.CollaborativeChannelAttention23(x2_nor, x3_nor)
        x1_CCAM, x2_CCAM_2 = self.CollaborativeChannelAttention12(x1_nor, x2_nor)
        x3_CCAM = self.CCAM_f3(x3_CCAM_1 + x3_CCAM_2)
        x2_CCAM = self.CCAM_f2(x2_CCAM_1 + x2_CCAM_2)


        # CSIM input: x1_CCAM x2_CCAM_update x3_CCAM_update x4_CCAM
        x1_CSAM, x2_CSAM_1 = self.CollaborativeSpatialAttention12(x1_CCAM, x2_CCAM)
        x2_CSAM_2, x3_CSAM_1 = self.CollaborativeSpatialAttention23(x2_CCAM, x3_CCAM)
        x3_CSAM_2, x4_CSAM = self.CollaborativeSpatialAttention34(x3_CCAM, x4_CCAM)
        x3_CSAM = self.CSAM_f3(x3_CSAM_1 + x3_CSAM_2)
        x2_CSAM = self.CSAM_f2(x2_CSAM_1 + x2_CSAM_2)

        x1_CS = x1_CSAM + x1_nor
        x2_CS = x2_CSAM + x2_nor
        x3_CS = x3_CSAM + x3_nor
        x4_CS = x4_CSAM + x4_nor

        #CPIM
        x3_SSA_1, x4_SSA = self.SpatialSelfAtt34(x3_CS, x4_CS)
        x2_SSA_1, x3_SSA_2 = self.SpatialSelfAtt23(x2_CS, x3_CS)
        x1_SSA, x2_SSA_2 = self.SpatialSelfAtt12(x1_CS, x2_CS)
        x3_SSA = self.SSA_f3(x3_SSA_1 + x3_SSA_2)
        x2_SSA = self.SSA_f2(x2_SSA_1 + x2_SSA_2)

        # Defect Reasoner
        s1, s2, s3, s4 = self.SalReasoner(x1_SSA, x2_SSA, x3_SSA, x4_SSA)
        s1 = self.upsample4(s1)
        s2 = self.upsample4(s2)
        s3 = self.upsample8(s3)
        s4 = self.upsample16(s4)

        return s1, s2, s3, s4, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4)

