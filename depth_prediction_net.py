import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1,):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def downsample_conv(in_channels, out_channels, stride=1):
    downsample = None
    if (in_channels == out_channels):
        if stride == 1:
            downsample = None
        else:
            downsample = nn.MaxPool2d(stride, stride)
    else:
        downsample = conv3x3(in_channels, out_channels, 1, stride, 0)
        
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
    layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

def conv(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3)


def upconv(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
                              stride=2, padding=1, output_padding=1)

def concat_and_pad(decoder_layer, encoder_layer):
    concat = torch.cat([decoder_layer, encoder_layer], axis=1)
    return F.pad(concat, (1, 1, 1, 1), mode='reflect')

class DispEncoder(nn.Module):
    def __init__(self, alpha=10, beta=0.01):
        super(DispEncoder, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [64, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(conv3x3(3, conv_planes[0], 7, 2, 3),
                                   nn.BatchNorm2d(conv_planes[0]),
                                   nn.ReLU(inplace=True))
        self.conv1_maxpool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1])
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2], 2)
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3], 2)
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4], 2)
        
        self.fc_layers = nn.Sequential(nn.Linear(256 * 8 * 8, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 11))
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv1_maxpool = self.conv1_maxpool(out_conv1)
        out_conv2 = self.conv2(out_conv1_maxpool)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        x = out_conv5.flatten(1)
        print(x.shape)
        x = self.fc_layers(x)
        return x, out_conv5, (out_conv4, out_conv3, out_conv2, out_conv1)

class DispNetS(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNetS, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [64, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(conv3x3(3, conv_planes[0], 7, 2, 3),
                                   nn.BatchNorm2d(conv_planes[0]),
                                   nn.ReLU(inplace=True))
        self.conv1_maxpool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1])
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2], 2)
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3], 2)
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4], 2)

        upconv_planes = [16, 32, 64, 128, 256]
        self.upconv5 = upconv(conv_planes[4],   upconv_planes[4])
        self.upconv4 = upconv(upconv_planes[4], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[2])
        self.upconv2 = upconv(upconv_planes[2], upconv_planes[1])
        self.upconv1 = upconv(upconv_planes[1], upconv_planes[0])

        self.iconv5 = conv(upconv_planes[4] + conv_planes[3], upconv_planes[4])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(upconv_planes[2] + conv_planes[1], upconv_planes[2])
        self.iconv2 = conv(upconv_planes[1] + conv_planes[0], upconv_planes[1])
        
        self.iconv1 = conv(upconv_planes[0], upconv_planes[0])
        self.iconv0 = conv(upconv_planes[0], 1)

        self.act = nn.Softplus()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv1_maxpool = self.conv1_maxpool(out_conv1)
        out_conv2 = self.conv2(out_conv1_maxpool)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        # print(out_conv4.shape)
        out_conv5 = self.conv5(out_conv4)
        # print(out_conv5.shape)

        out_upconv5 = self.upconv5(out_conv5)
        # print(out_upconv5.shape)
        concat5 = concat_and_pad(out_upconv5, out_conv4)
        # print(concat5.shape)
        out_iconv5 = self.iconv5(concat5)
        # print(out_iconv5.shape)

        out_upconv4 = self.upconv4(out_conv4)
        # print(out_upconv4.shape)
        concat4 = concat_and_pad(out_upconv4, out_conv3)
        # print(concat4.shape)
        out_iconv4 = self.iconv4(concat4)
        # print(out_iconv4.shape)

        out_upconv3 = self.upconv3(out_conv3)
        # print(out_upconv3.shape)
        concat3 = concat_and_pad(out_upconv3, out_conv2)
        # print(concat3.shape)
        out_iconv3 = self.iconv3(concat3)
        # print(out_iconv3.shape)

        out_upconv2 = self.upconv2(out_conv2)
        # print(out_upconv2.shape)
        concat2 = concat_and_pad(out_upconv2, out_conv1)
        # print(concat2.shape)
        out_iconv2 = self.iconv2(concat2)
        # print(out_iconv2.shape)

        out_upconv1 = self.upconv1(out_iconv2)
        out_upconv1 = F.pad(out_upconv1, (1, 1, 1, 1), mode='reflect')

        out_iconv1 = self.iconv1(out_upconv1)
        
        out_upconv0 = F.pad(out_iconv1, (1, 1, 1, 1), mode='reflect')
        out_iconv0 = self.iconv0(out_upconv0)
        
        depth = self.act(out_iconv0)
        
        return depth

if __name__ == "__main__":
    x = torch.randn((1,3,160,320))
    x = x.to(device='cuda')
    m = DispNetS()
    m = m.to(device='cuda')
    o = m(x)
    print(o.shape)