import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            if torch.cuda.is_available():
                mask = mask.cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero(as_tuple=False)
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), 1)
        if torch.cuda.is_available():
            offsets = offsets.cuda()

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5,
                 drop_block=True, **kwargs):
        self.inplanes = 3
        # print(drop_block)
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=drop_block,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=drop_block,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def Res12(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


class ConvNet(nn.Module):
    """
    Conv4 Backbone
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 64

        # set layers
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=self.hidden,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden,
                            out_channels=self.hidden,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv_3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden,
                            out_channels=self.hidden,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv_4 = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden,
                            out_channels=self.hidden,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        out = output_data.view(output_data.size(0), -1)
        return out


# def conv_block(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )
#
#
# # class ConvNet(nn.Module):
# #
# #     def __init__(self, x_dim=3, hid_dim=64, z_dim=64, pool=True):
# #         super().__init__()
# #         # self.encoder = nn.Sequential(
# #         #     conv_block(x_dim, hid_dim),
# #         #     conv_block(hid_dim, hid_dim),
# #         #     conv_block(hid_dim, hid_dim),
# #         #     conv_block(hid_dim, z_dim),
# #         # )
# #         self.encoder = nn.ModuleList([
# #             conv_block(x_dim, hid_dim),
# #             conv_block(hid_dim, hid_dim),
# #             conv_block(hid_dim, hid_dim),
# #             conv_block(hid_dim, z_dim),
# #         ])
# #         self.pool = pool
# #
# #     def forward(self, x):
# #
# #         for l in self.encoder:
# #             x = l(x)
# #         if self.pool:
# #             x = F.adaptive_avg_pool2d(x, 1)
# #         x = x.view(x.size(0), -1)
# #         return x
# #
# #     def pre_forward(self, x, layer=None):
# #         if layer is None:
# #             return self.forward(x)
# #         if layer == 0:
# #             return x
# #         for i in range(layer):
# #             x = self.encoder[i](x)
# #         return x
# #
# #     def post_forward(self, x, layer=None):
# #         if layer is None:
# #             return x
# #         if layer < len(self):
# #             for i in range(layer, len(self)):
# #                 x = self.encoder[i](x)
# #         if self.pool:
# #             x = F.adaptive_avg_pool2d(x, 1)
# #         x = x.view(x.size(0), -1)
# #         return x
# #
# #     def __len__(self):
# #         return len(self.encoder)

if __name__ == '__main__':
    # params = {}
    # params['drop_block'] = False
    # net=ResNet(**params).cuda(0)
    # print(net)
    net = ConvNet()
    img = torch.randn([1, 3, 32, 32])
    a = net(img)
    print(a.size())


