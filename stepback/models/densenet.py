"""
This script has been taken from: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py
We added the option to remove BatchNorm. 
We use the model for CIFAR100, only changing the dimension of the last linear layer.
    Reference:
    [1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

        Densely Connected Convolutional Networks
        https://arxiv.org/abs/1608.06993v5
    [2] https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
    [3] https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py
    If you use this implementation in you work, please don't forget to mention the
    author, .
"""
import torch
import torch.nn as nn

def get_num_groups(num_channels, max_groups=8):
    for num_groups in reversed(range(1, max_groups + 1)):
        if num_channels % num_groups == 0:
            return num_groups
    return 1  # fallback to LayerNorm-like

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, norm='batch_norm'):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        if 'batch_norm' in norm:        
            norm1 = nn.BatchNorm2d(in_channels)
            norm2 = nn.BatchNorm2d(inner_channel)
        elif 'group_norm' in norm:
            num_groups1 = get_num_groups(in_channels)
            num_groups2 = get_num_groups(inner_channel)

            norm1 = nn.GroupNorm(num_groups1, in_channels, affine=True)
            norm2 = nn.GroupNorm(num_groups2, inner_channel, affine=True)
        else:
            norm1 = lambda x: x
            norm2 = lambda x: x

        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            norm2,
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch_norm'):
        super().__init__()

        if 'batch_norm' in norm:        
            norm1 = nn.BatchNorm2d(in_channels)
        elif 'group_norm' in norm:
            num_groups = get_num_groups(in_channels)

            norm1 = nn.GroupNorm(num_groups, in_channels, affine=True)
        else:
            norm1 = lambda x: x

        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            norm1,
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100, norm='batch_norm'):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        self.norm = norm

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels, norm=norm))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]

        if 'batch_norm' in norm:        
            norm1 = nn.BatchNorm2d(inner_channels)
        elif 'group_norm' in norm:
            num_groups = get_num_groups(inner_channels)
            norm1 = nn.GroupNorm(num_groups, inner_channels, affine=True)
        else:
            norm1 = lambda x: x

        #"""We find this design especially effective for DenseNet and
        self.features.add_module('bn', norm1)
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate, norm=self.norm))
            in_channels += self.growth_rate
        return dense_block

def densenet100(norm='batch_norm'):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, norm=norm)

def densenet121(norm='batch_norm'):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, norm=norm)

def densenet169(norm='batch_norm'):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, norm=norm)

def densenet201(norm='batch_norm'):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, norm=norm)

def densenet161(norm='batch_norm'):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, norm=norm)

def _get_densenet(name, norm='batch_norm'):
    
    if name == 'densenet100':
        m = densenet100(norm=norm)
    elif name == 'densenet121':
        m = densenet121(norm=norm)
    elif name == 'densenet169':
        m = densenet169(norm=norm)
    elif name == 'densenet201':
        m = densenet201(norm=norm)
    elif name == 'densenet161':
        m = densenet161(norm=norm)
    return m

def get_cifar_densenet(name, norm='batch_norm'):
    m = _get_densenet(name, norm)
    return m