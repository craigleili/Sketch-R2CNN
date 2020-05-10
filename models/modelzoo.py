from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pretrainedmodels


class CNNBackbone(torch.nn.Module):

    def __init__(self, pretrained=True, requires_grad=True, in_channels=3):
        super().__init__()

        self.pretrained = pretrained
        self.requires_grad = requires_grad
        self.in_channels = in_channels

        self.num_out_features = self._init()

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _init(self):
        raise NotImplementedError

    def forward(self, *x):
        raise NotImplementedError


class DenseNet161Backbone(CNNBackbone):

    def _init(self):
        cnn = torchvision.models.densenet161(pretrained=self.pretrained)

        if self.in_channels in [1, 3]:
            self.features = cnn.features
            print('[*] DenseNet161Backbone: use pretrained conv0 with {} input channels'.format(
                self.features.conv0.in_channels))
        else:
            from collections import OrderedDict
            print('[*] DenseNet161Backbone: cnn.features -', cnn.features.__class__.__name__)
            module_dict = OrderedDict()
            for name, module in cnn.features.named_children():
                if name == 'conv0':
                    module_dict[name + '_new'] = nn.Conv2d(self.in_channels,
                                                           module.out_channels,
                                                           kernel_size=module.kernel_size,
                                                           stride=module.stride,
                                                           padding=module.padding,
                                                           bias=False)
                else:
                    module_dict[name] = module
            self.features = nn.Sequential(module_dict)
            print('[*] DenseNet161Backbone: use a new conv0 with {} input channels'.format(self.in_channels))

        num_out_features = cnn.classifier.in_features
        return num_out_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        return out


class ResNet101Backbone(CNNBackbone):

    def _init(self):
        cnn = torchvision.models.resnet101(pretrained=self.pretrained)

        if self.in_channels in [1, 3]:
            self.conv1 = cnn.conv1
            self.conv1_new = None
            print('[*] ResNet101Backbone: use pretrained conv1 with {} input channels'.format(cnn.conv1.in_channels))
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = None
            print('[*] ResNet101Backbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        self.avgpool = cnn.avgpool

        num_out_features = 512 * 4
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class ResNet50Backbone(CNNBackbone):
    def _init(self):
        cnn = torchvision.models.resnet50(pretrained=self.pretrained)
        if self.in_channels in [1, 3]:
            self.conv1 = cnn.conv1
            self.conv1_new = None
            print('[*] ResNet50Backbone: use pretrained conv1 with {} input channels'.format(cnn.conv1.in_channels))
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = None
            print('[*] ResNet50Backbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        self.avgpool = cnn.avgpool

        num_out_features = 512 * 4
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class SketchANetBackbone(CNNBackbone):

    def _init(self):
        if self.in_channels in [1, 3]:
            self.conv1 = nn.Conv2d(3, 64, 15, stride=3, padding=0)
            self.conv1_new = None
            print('[*] SketchANetBackbone: use conv1 with 3 input channels')
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, 15, stride=3, padding=0)
            self.conv1 = None
            print('[*] SketchANetBackbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 512)

        num_out_features = 512
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0) 

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0) 

        x = self.conv3(x)
        x = F.relu(x) 

        x = self.conv4(x)
        x = F.relu(x) 

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0) 

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


CNN_MODELS = {
    'densenet161': DenseNet161Backbone,
    'resnet101': ResNet101Backbone,
    'resnet50': ResNet50Backbone,
    'sketchanet': SketchANetBackbone,
}

CNN_IMAGE_SIZES = {
    'densenet161': 224,
    'resnet101': 224,
    'resnet50': 224,
    'sketchanet': 225,
}
