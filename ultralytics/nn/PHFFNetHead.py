import copy
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ['PHFFNet4Head']


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2),
                                         self.upsample_scalethree3_8(x3))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2),
                                       self.upsample_scalethree3_4(x3))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2,
                                       self.upsample_scalethree3_2(x3))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1),
                                           self.downsample_scaletwo3_2(x2), x3)

        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        return scalezero, scaleone, scaletwo, scalethree


class PHFFNet(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=128):
        super(PHFFNet, self).__init__()

        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        return out0, out1, out2, out3


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class PHFFNet4Head(nn.Module):
    """YOLOv8 Detect head for detection models. CSDNSnu77"""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, channel=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(channel, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(channel, channel, 3), Conv(channel, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.PHFFNet = PHFFNet(ch, channel)  # 4头

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        x = list(self.PHFFNet(x))
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.
        Args:
            x (tensor): Input tensor.
        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.
        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.
        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

# import math
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ultralytics.nn.modules import DFL
# from ultralytics.nn.modules.conv import Conv
# from ultralytics.utils.tal import dist2bbox, make_anchors
#
# __all__ = ['PHFFNet4Head']
#
#
# def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
#     if not pad:
#         pad = (kernel_size - 1) // 2 if kernel_size else 0
#     else:
#         pad = pad
#     return nn.Sequential(OrderedDict([
#         ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
#         ("bn", nn.BatchNorm2d(filter_out)),
#         ("relu", nn.ReLU(inplace=True)),
#     ]))
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, filter_in, filter_out):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Upsample(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super(Upsample, self).__init__()
#
#         self.upsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 1),
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear')
#         )
#
#     def forward(self, x):
#         x = self.upsample(x)
#
#         return x
#
#
# class Downsample_x2(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Downsample_x2, self).__init__()
#
#         self.downsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 2, 2, 0)
#         )
#
#     def forward(self, x, ):
#         x = self.downsample(x)
#
#         return x
#
#
# class Downsample_x4(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Downsample_x4, self).__init__()
#
#         self.downsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 4, 4, 0)
#         )
#
#     def forward(self, x, ):
#         x = self.downsample(x)
#
#         return x
#
#
# class Downsample_x8(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Downsample_x8, self).__init__()
#
#         self.downsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 8, 8, 0)
#         )
#
#     def forward(self, x, ):
#         x = self.downsample(x)
#
#         return x
#
#
# class ASFF_2(nn.Module):
#     def __init__(self, inter_dim=512):
#         super(ASFF_2, self).__init__()
#
#         self.inter_dim = inter_dim
#         compress_c = 8
#
#         self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
#
#         self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)
#
#     def forward(self, input1, input2):
#         level_1_weight_v = self.weight_level_1(input1)
#         level_2_weight_v = self.weight_level_2(input2)
#
#         levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
#                             input2 * levels_weight[:, 1:2, :, :]
#
#         out = self.conv(fused_out_reduced)
#
#         return out
#
#
# class ASFF_3(nn.Module):
#     def __init__(self, inter_dim=512):
#         super(ASFF_3, self).__init__()
#
#         self.inter_dim = inter_dim
#         compress_c = 8
#
#         self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
#
#         self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)
#
#     def forward(self, input1, input2, input3):
#         level_1_weight_v = self.weight_level_1(input1)
#         level_2_weight_v = self.weight_level_2(input2)
#         level_3_weight_v = self.weight_level_3(input3)
#
#         levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
#                             input2 * levels_weight[:, 1:2, :, :] + \
#                             input3 * levels_weight[:, 2:, :, :]
#
#         out = self.conv(fused_out_reduced)
#
#         return out
#
#
# class ASFF_4(nn.Module):
#     def __init__(self, inter_dim=512):
#         super(ASFF_4, self).__init__()
#
#         self.inter_dim = inter_dim
#         compress_c = 8
#
#         self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
#
#         self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)
#
#     def forward(self, input0, input1, input2, input3):
#         level_0_weight_v = self.weight_level_0(input0)
#         level_1_weight_v = self.weight_level_1(input1)
#         level_2_weight_v = self.weight_level_2(input2)
#         level_3_weight_v = self.weight_level_3(input3)
#
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
#                             input1 * levels_weight[:, 1:2, :, :] + \
#                             input2 * levels_weight[:, 2:3, :, :] + \
#                             input3 * levels_weight[:, 3:, :, :]
#
#         out = self.conv(fused_out_reduced)
#
#         return out
#
#
# class BlockBody(nn.Module):
#     def __init__(self, channels=[64, 128, 256, 512]):
#         super(BlockBody, self).__init__()
#
#         self.blocks_scalezero1 = nn.Sequential(
#             BasicConv(channels[0], channels[0], 1),
#         )
#         self.blocks_scaleone1 = nn.Sequential(
#             BasicConv(channels[1], channels[1], 1),
#         )
#         self.blocks_scaletwo1 = nn.Sequential(
#             BasicConv(channels[2], channels[2], 1),
#         )
#         self.blocks_scalethree1 = nn.Sequential(
#             BasicConv(channels[3], channels[3], 1),
#         )
#
#         self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
#         self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)
#
#         self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
#         self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])
#
#         self.blocks_scalezero2 = nn.Sequential(
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#         )
#         self.blocks_scaleone2 = nn.Sequential(
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#         )
#
#         self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
#         self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
#         self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
#         self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
#         self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
#         self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)
#
#         self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
#         self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
#         self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])
#
#         self.blocks_scalezero3 = nn.Sequential(
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#         )
#         self.blocks_scaleone3 = nn.Sequential(
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#         )
#         self.blocks_scaletwo3 = nn.Sequential(
#             BasicBlock(channels[2], channels[2]),
#             BasicBlock(channels[2], channels[2]),
#             BasicBlock(channels[2], channels[2]),
#             BasicBlock(channels[2], channels[2]),
#         )
#
#         self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
#         self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
#         self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
#         self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
#         self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
#         self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
#         self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
#         self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
#         self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
#         self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
#         self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
#         self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)
#
#         self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
#         self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
#         self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
#         self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])
#
#         self.blocks_scalezero4 = nn.Sequential(
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#             BasicBlock(channels[0], channels[0]),
#         )
#         self.blocks_scaleone4 = nn.Sequential(
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#             BasicBlock(channels[1], channels[1]),
#         )
#         self.blocks_scaletwo4 = nn.Sequential(
#             BasicBlock(channels[2], channels[2]),
#             BasicBlock(channels[2], channels[2]),
#             BasicBlock(channels[2], channels[2]),
#             BasicBlock(channels[2], channels[2]),
#         )
#         self.blocks_scalethree4 = nn.Sequential(
#             BasicBlock(channels[3], channels[3]),
#             BasicBlock(channels[3], channels[3]),
#             BasicBlock(channels[3], channels[3]),
#             BasicBlock(channels[3], channels[3]),
#         )
#
#     def forward(self, x):
#         x0, x1, x2, x3 = x
#
#         x0 = self.blocks_scalezero1(x0)
#         x1 = self.blocks_scaleone1(x1)
#         x2 = self.blocks_scaletwo1(x2)
#         x3 = self.blocks_scalethree1(x3)
#
#         scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
#         scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)
#
#         x0 = self.blocks_scalezero2(scalezero)
#         x1 = self.blocks_scaleone2(scaleone)
#
#         scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
#         scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
#         scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)
#
#         x0 = self.blocks_scalezero3(scalezero)
#         x1 = self.blocks_scaleone3(scaleone)
#         x2 = self.blocks_scaletwo3(scaletwo)
#
#         scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2),
#                                          self.upsample_scalethree3_8(x3))
#         scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2),
#                                        self.upsample_scalethree3_4(x3))
#         scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2,
#                                        self.upsample_scalethree3_2(x3))
#         scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1),
#                                            self.downsample_scaletwo3_2(x2), x3)
#
#         scalezero = self.blocks_scalezero4(scalezero)
#         scaleone = self.blocks_scaleone4(scaleone)
#         scaletwo = self.blocks_scaletwo4(scaletwo)
#         scalethree = self.blocks_scalethree4(scalethree)
#
#         return scalezero, scaleone, scaletwo, scalethree
#
#
# class PHFFNet(nn.Module):
#     def __init__(self,
#                  in_channels=[256, 512, 1024, 2048],
#                  out_channels=128):
#         super(PHFFNet, self).__init__()
#
#         self.fp16_enabled = False
#
#         self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
#         self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
#         self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
#         self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)
#
#         self.body = nn.Sequential(
#             BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
#         )
#
#         self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
#         self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
#         self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
#         self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
#         self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)
#
#         # init weight
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight, gain=0.02)
#             elif isinstance(m, nn.BatchNorm2d):
#                 torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#                 torch.nn.init.constant_(m.bias.data, 0.0)
#
#     def forward(self, x):
#         x0, x1, x2, x3 = x
#
#         x0 = self.conv0(x0)
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)
#         x3 = self.conv3(x3)
#
#         out0, out1, out2, out3 = self.body([x0, x1, x2, x3])
#
#         out0 = self.conv00(out0)
#         out1 = self.conv11(out1)
#         out2 = self.conv22(out2)
#         out3 = self.conv33(out3)
#
#         return out0, out1, out2, out3
#
#
# class PHFFNet4Head(nn.Module):
#     """YOLOv8 Detect head for detection models."""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#
#     def __init__(self, nc=80, channel=128, ch=()):
#         """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(channel, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(Conv(channel, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#         self.PHFFNet = PHFFNet(ch)
#
#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         x = list(self.PHFFNet(x))
#         shape = x[0].shape  # BCHW
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:
#             return x
#         elif self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
#             box = x_cat[:, :self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
#
#         if self.export and self.format in ('tflite', 'edgetpu'):
#             # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
#             # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
#             # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
#             img_h = shape[2] * self.stride[0]
#             img_w = shape[3] * self.stride[0]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
#             dbox /= img_size
#
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)
#
#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
