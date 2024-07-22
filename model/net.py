import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_mix(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=False,
                                   dilation=(dilated, 1))

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=False,
                                   dilation=(1, dilated))

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv5x1_2 = nn.Conv2d(chann, chann, (5, 1), stride=1, padding=(2 * dilated, 0), bias=False,
                                   dilation=(dilated, 1))

        self.conv1x5_2 = nn.Conv2d(chann, chann, (1, 5), stride=1, padding=(0, 2 * dilated), bias=False,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv7x1_2 = nn.Conv2d(chann, chann, (7, 1), stride=1, padding=(3 * dilated, 0), bias=False,
                                   dilation=(dilated, 1))

        self.conv1x7_2 = nn.Conv2d(chann, chann, (1, 7), stride=1, padding=(0, 3 * dilated), bias=False,
                                   dilation=(1, dilated))

        self.bn3 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output + input)

        output = self.conv5x1_2(output)
        output = F.relu(output)
        output = self.conv1x5_2(output)
        output = self.bn2(output)
        output = F.relu(output + input)

        output = self.conv7x1_2(output)
        output = F.relu(output)
        output = self.conv1x7_2(output)
        output = self.bn3(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = self.attention(output + input)

        return F.relu(output)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class FeatureFlipModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFlipModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # 假设输入 x 的形状为 (batch_size, channels, height, width)

        # 垂直翻转
        vertical_flip = torch.flip(x, dims=[2])

        # 水平翻转
        horizontal_flip = torch.flip(x, dims=[3])

        # 对垂直翻转的特征进行卷积处理
        vertical_flip = self.conv1(vertical_flip)
        vertical_flip = self.conv2(vertical_flip)

        # 对水平翻转的特征进行卷积处理
        horizontal_flip = self.conv1(horizontal_flip)
        horizontal_flip = self.conv2(horizontal_flip)

        # 将原始特征和翻转后的特征进行拼接
        flipped_features = torch.cat([x, vertical_flip, horizontal_flip], dim=1)
        flipped_features = self.conv3(flipped_features)

        return flipped_features


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        self.layers.append(conv_mix(64, 0.1, 1))
        self.layers.append(conv_mix(64, 0.1, 1))
        self.layers.append(conv_mix(64, 0.1, 1))
        self.layers.append(conv_mix(64, 0.1, 1))
        self.layers.append(conv_mix(64, 0.1, 1))
        self.flip_1 = FeatureFlipModule(in_channels=64, out_channels=64)

        self.layers.append(DownsamplerBlock(64, 128))

        self.layers.append(conv_mix(128, 0.1, 1))
        self.layers.append(conv_mix(128, 0.1, 2))
        self.layers.append(conv_mix(128, 0.1, 3))
        self.layers.append(conv_mix(128, 0.1, 5))
        self.flip_2 = FeatureFlipModule(in_channels=128, out_channels=128)

        self.layers.append(conv_mix(128, 0.1, 1))
        self.layers.append(conv_mix(128, 0.1, 2))
        self.layers.append(conv_mix(128, 0.1, 3))
        self.layers.append(conv_mix(128, 0.1, 5))
        self.flip_3 = FeatureFlipModule(in_channels=128, out_channels=128)

    def forward(self, input):  # [batch,3,288,800]
        x = self.initial_block(input)  # [batch,16,144,400]
        x1 = self.layers[0](x)  # x1[batch,64,72,200]

        x = self.layers[1](x1)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x2 = self.layers[5](x)  # x2[batch,64,72,200]
        x = self.flip_1(x2)

        x = self.layers[6](x)
        x = self.layers[7](x)
        x = self.layers[8](x)
        x = self.layers[9](x)
        x3 = self.layers[10](x)  # x3[batch,128,36,100]
        x = self.flip_2(x3)

        x = self.layers[11](x)
        x = self.layers[12](x)
        x = self.layers[13](x)
        x4 = self.layers[14](x)  # x4[batch,128,36,100]
        x5 = self.flip_3(x4)

        return x1, x2, x3, x4, x5


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, up_width, up_height):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)
        self.follows = nn.ModuleList()
        self.follows.append(conv_mix(noutput, 0, 1))
        self.follows.append(conv_mix(noutput, 0, 1))

        # interpolate
        self.up_width = up_width
        self.up_height = up_height
        self.interpolate_conv = conv1x1(ninput, noutput)
        self.interpolate_bn = nn.BatchNorm2d(
            noutput, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        out = F.relu(output)
        for follow in self.follows:
            out = follow(out)

        interpolate_output = self.interpolate_conv(input)
        interpolate_output = self.interpolate_bn(interpolate_output)
        interpolate_output = F.relu(interpolate_output)

        interpolate = F.interpolate(interpolate_output, size=[self.up_height, self.up_width],
                                    mode='bilinear', align_corners=False)

        return out + interpolate


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        img_height = 288
        img_width = 800
        num_classes = num_classes

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(ninput=128, noutput=64,
                                          up_height=int(img_height) // 4, up_width=int(img_width) // 4))
        self.layers.append(UpsamplerBlock(ninput=64, noutput=32,
                                          up_height=int(img_height) // 2, up_width=int(img_width) // 2))
        self.layers.append(UpsamplerBlock(ninput=32, noutput=16,
                                          up_height=int(img_height) // 1, up_width=int(img_width) // 1))

        self.output_conv = conv1x1(16, num_classes)

    def forward(self, input):
        output_1 = self.layers[0](input)
        output_2 = self.layers[1](output_1)
        output = self.layers[2](output_2)
        output = self.output_conv(output)

        return output_1, output_2, output


class Aux_Seg(nn.Module):
    def __init__(self, ninput, noutput, rate):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput, 1, bias=False)
        self.up = nn.Upsample(scale_factor=rate, mode='bilinear', align_corners=False)

    def forward(self, input):
        output = self.conv(input)
        output = self.up(output)
        return output


class SPA(nn.Module):
    def __init__(self, num_classes):
        super(SPA, self).__init__()

        self.conv1 = (nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1))
        self.bn1 = (nn.BatchNorm2d(128, eps=1e-03))
        self.conv2 = (
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=(4, 4), bias=False,
                      dilation=(4, 4)))
        self.bn2 = (nn.BatchNorm2d(128, eps=1e-03))
        self.conv3 = (
            nn.Conv2d(in_channels=128, out_channels=num_classes - 1, kernel_size=(1, 1), stride=1, padding=(0, 0),
                      bias=False))
        self.bn3 = (nn.BatchNorm2d(num_classes - 1, eps=1e-03))
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x1 = self.pool1(x)
        x1 = torch.sigmoid(x1)
        x1 = x1.expand(-1, 4, 36, 100)
        x1 = torch.nn.functional.interpolate(x1, scale_factor=8, mode='bilinear')

        x2 = self.pool2(x)
        x2 = torch.sigmoid(x2)
        x2 = x2.expand(-1, 4, 36, 100)
        x2 = torch.nn.functional.interpolate(x2, scale_factor=8, mode='bilinear')

        return x1, x2


class Net(nn.Module):
    def __init__(self, num_classes):  # use encoder to pass pretrained encoder
        super().__init__()
        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)
        self.aux_2 = Aux_Seg(64, 5, 4)
        self.aux_3 = Aux_Seg(32, 5, 2)
        self.esa = SPA(num_classes=num_classes)

    def forward(self, input):
        x1, x2, x3, x4, x5 = self.encoder(input)
        x1 = torch.nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x2 = torch.nn.functional.interpolate(x2, scale_factor=0.5, mode='bilinear')
        cat_x_1 = torch.cat([x1, x2, x3, x4], dim=1)
        att_out_1, att_out_2 = self.esa(cat_x_1)
        seg_out_1, seg_out_2, seg_out = self.decoder(x5)
        aux_2 = self.aux_2(seg_out_1)
        aux_3 = self.aux_3(seg_out_2)
        return seg_out, att_out_1, att_out_2, aux_2, aux_3



if __name__ == "__main__":
    test = torch.rand(4, 3, 288, 800)
    net = Net(num_classes=5)
    # print(net)

    # 开始计时
    start_time = time.time()

    # 执行前向传播
    seg_out, att_out_1, att_out_2, aux_2, aux_3 = net(test)
    print(seg_out.shape)
    print(att_out_1.shape)
    print(att_out_2.shape)
    print(aux_2.shape)
    print(aux_3.shape)

    # 结束计时
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    # 打印结果
    print("代码执行时间：", execution_time, "秒")

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"total_params：{total_params}")
    print(f"trainable_params：{trainable_params}")
