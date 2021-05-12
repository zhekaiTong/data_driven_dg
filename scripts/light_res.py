import torch
import torch.nn as nn

# Construct customized ResNet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class myResNet(nn.Module):

    def __init__(self, pcpt_block, pcpt_layers, dg_block, dg_layers, dg_h, dg_w, dg_is_upsample=0):
        self.inplanes = 64
        #self.pcpt_is_upsample = pcpt_is_upsample
        super(myResNet, self).__init__()
        self.pcpt_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.pcpt_bn1 = nn.BatchNorm2d(64)
        self.pcpt_relu = nn.ReLU(inplace=True)
        self.pcpt_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pcpt_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.pcpt_layer1 = self._make_layer(pcpt_block, 128, pcpt_layers[0])
        self.pcpt_layer2 = self._make_layer(pcpt_block, 256, pcpt_layers[1])
        self.pcpt_layer3 = self._make_layer(pcpt_block, 512, pcpt_layers[2])

        self.inplanes = 512
        self.dg_is_upsample = dg_is_upsample
        self.dg_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dg_layer1 = self._make_layer(dg_block, 256, dg_layers[0])
        self.dg_layer2 = self._make_layer(dg_block, 128, dg_layers[1])
        self.dg_layer3 = self._make_layer(dg_block, 64, dg_layers[2])
        self.dg_conv1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.dg_bn1 = nn.BatchNorm2d(1)
        self.dg_conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.dg_bn2 = nn.BatchNorm2d(3)
        self.dg_relu = nn.ReLU(inplace=True)

        self.dg_head = nn.Linear(dg_h*dg_w, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pcpt_conv1(x)
        x = self.pcpt_bn1(x)
        x = self.pcpt_relu(x)
        x = self.pcpt_maxpool(x)

        x = self.pcpt_layer1(x)
        x = self.pcpt_maxpool(x)
        x = self.pcpt_layer2(x)
        x = self.pcpt_layer3(x)

        x = self.dg_layer1(x)
        x = self.dg_layer2(x)
        x = self.dg_upsample(x)
        x = self.dg_layer3(x)
        x = self.dg_upsample(x)

        x = self.dg_conv2(x)
        x = self.dg_bn2(x)
        x = self.dg_relu(x)

        return x
