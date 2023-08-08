# ResNet for CIFAR (32x32)
# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
from typing import Sequence

class DropLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.dummy_weights = torch.eye(self.in_channels).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        self.dummy_weights = self.dummy_weights.to(x.device)
        return F.conv2d(x, self.dummy_weights)
    
    def __repr__(self):
        return f"DropLayer(in_channels={self.in_channels}, out_channels={self.out_channels})"

class DropLayerPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: DropLayer, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels -= len(idxs)
        layer.dummy_weights = torch.index_select(layer.dummy_weights, 0, torch.LongTensor(keep_idxs).to(layer.dummy_weights.device)).to(layer.dummy_weights.device)
        return layer
    
    def get_out_channels(self, layer):
        return layer.out_channels
    
    def prune_in_channels(self, layer: DropLayer, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        layer.in_channels -= len(idxs)
        layer.dummy_weights = torch.index_select(layer.dummy_weights, 1, torch.LongTensor(keep_idxs).to(layer.dummy_weights.device)).to(layer.dummy_weights.device)
        return layer

    def get_in_channels(self, layer):
        return layer.in_channels

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        else:
            # self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            # self.shortcut.weight.data = torch.eye(in_planes).unsqueeze(-1).unsqueeze(-1)
            # self.shortcut.requires_grad_(False)
            self.shortcut = DropLayer(in_planes)
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        else:
            # self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            # self.shortcut.weight.data = torch.eye(in_planes).unsqueeze(-1).unsqueeze(-1)
            # self.shortcut.requires_grad_(False)
            self.shortcut = DropLayer(in_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        if return_features:
            return out, feature
        else:
            return out
 
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
 
def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)