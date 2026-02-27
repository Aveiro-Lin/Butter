import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd  # kept for compatibility, even if unused


# -------------------------
# Helpers
# -------------------------
def autopad(k, p=None):  # kernel, padding
    """Pad to 'same' (matches original behavior)."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# -------------------------
# Basic blocks
# -------------------------
class Conv(nn.Module):
    # Standard convolution (matches original behavior)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        return self.downsample(x)


# -------------------------
# ASFF modules
# -------------------------
class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[256, 512]):
        super().__init__()

        self.inter_dim = inter_dim
        self.level = level
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])

    def _align(self, input1, input2):
        # identical alignment logic to the original forward()
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)
        return input1, input2

    def forward(self, x):
        input1, input2 = x
        input1, input2 = self._align(input1, input2)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :]
        return self.conv(fused_out_reduced)


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super().__init__()

        self.inter_dim = inter_dim
        self.level = level
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        # keep exact module construction patterns
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)

    def _align(self, input1, input2, input3):
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        return input1, input2, input3

    def forward(self, x):
        input1, input2, input3 = x
        input1, input2, input3 = self._align(input1, input2, input3)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)

        fused_out_reduced = (
            input1 * levels_weight[:, 0:1, :, :]
            + input2 * levels_weight[:, 1:2, :, :]
            + input3 * levels_weight[:, 2:, :, :]
        )
        return self.conv(fused_out_reduced)


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[16, 32, 64, 128]):
        super().__init__()

        self.inter_dim = inter_dim
        self.level = level
        compress_c = 8

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        # keep exact module construction patterns
        if self.level == 0:
            self.upsample8x = Upsample(channel[3], channel[0], scale_factor=8)
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample4x1 = Upsample(channel[3], channel[1], scale_factor=4)
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.upsample2x2 = Upsample(channel[3], channel[2], scale_factor=2)
            self.downsample2x2 = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x2 = Downsample(channel[0], channel[2], scale_factor=4)
        elif self.level == 3:
            self.downsample2x3 = Downsample(channel[2], channel[3], scale_factor=2)
            self.downsample4x3 = Downsample(channel[1], channel[3], scale_factor=4)
            self.downsample8x3 = Downsample(channel[0], channel[3], scale_factor=8)

    def _align(self, input1, input2, input3, input4):
        # identical alignment logic to the original forward()
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            input4 = self.upsample8x(input4)
        elif self.level == 1:
            input1 = self.downsample2x1(input1)
            input3 = self.upsample2x1(input3)
            input4 = self.upsample4x1(input4)
        elif self.level == 2:
            input1 = self.downsample4x2(input1)
            input2 = self.downsample2x2(input2)
            input4 = self.upsample2x2(input4)
        elif self.level == 3:
            input1 = self.downsample8x3(input1)
            input2 = self.downsample4x3(input2)
            input3 = self.downsample2x3(input3)
        return input1, input2, input3, input4

    def forward(self, x):
        input1, input2, input3, input4 = x
        input1, input2, input3, input4 = self._align(input1, input2, input3, input4)

        level_0_weight_v = self.weight_level_0(input1)
        level_1_weight_v = self.weight_level_1(input2)
        level_2_weight_v = self.weight_level_2(input3)
        level_3_weight_v = self.weight_level_3(input4)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)

        fused_out_reduced = (
            input1 * levels_weight[:, 0:1, :, :]
            + input2 * levels_weight[:, 1:2, :, :]
            + input3 * levels_weight[:, 2:3, :, :]
            + input4 * levels_weight[:, 3:, :, :]
        )
        return self.conv(fused_out_reduced)


if __name__ == '__main__':
    input_0 = torch.randn(1, 16, 80, 80)
    input_1 = torch.randn(1, 32, 40, 40)
    input_2 = torch.randn(1, 64, 20, 20)
    input_3 = torch.randn(1, 128, 10, 10)
    input = [input_0, input_1, input_2, input_3]

    model1 = ASFF_4(128, level=3, channel=[16, 32, 64, 128])
    output = model1(input)
    print(output.shape)