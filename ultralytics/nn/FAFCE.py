import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint 


# -------------------------
# Init helpers 
# -------------------------
def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# -------------------------
# Small utilities 
# -------------------------
def resize(input,
           size=None,
           scale_factor=None,
           mode: str = 'nearest',
           align_corners=None,
           warning: bool = True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                        and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M: int, N: int):
    """Generate a 2D Hamming window via outer product."""
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


# -------------------------
# Core ops 
# -------------------------
def carafe(x: torch.Tensor,
           normed_mask: torch.Tensor,
           kernel_size: int,
           group: int = 1,
           up: int = 1) -> torch.Tensor:
    b, c, h, w = x.shape
    _, _, m_h, m_w = normed_mask.shape

    assert m_h == up * h
    assert m_w == up * w

    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)

    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)

    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    out = (unfold_x * normed_mask).sum(dim=2).reshape(b, c, m_h, m_w)
    return out


def compute_similarity(input_tensor: torch.Tensor, k: int = 3, dilation: int = 1, sim: str = 'cos') -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

    if sim == 'cos':
        similarity = F.cosine_similarity(
            unfold_tensor[:, :, k * k // 2:k * k // 2 + 1],
            unfold_tensor[:, :, :],
            dim=1
        )
    elif sim == 'dot':
        similarity = (unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]).sum(dim=1)
    else:
        raise NotImplementedError

    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
    return similarity.view(B, k * k - 1, H, W)


class LocalSimGuidedSampler(nn.Module):
    """
    offset generator in FAFCE
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, use_direct_scale=True,
                 kernel_size=1, local_window=3, sim_type='cos', norm=True, direction_feat='sim_concat'):
        super().__init__()
        assert scale == 2
        assert style == 'lp'

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(local_window ** 2 - 1, out_channels,
                                    kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == 'sim_concat':
            self.offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels,
                                    kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale = nn.Conv2d(in_channels, out_channels,
                                              kernel_size=kernel_size, padding=kernel_size // 2)
            elif self.direction_feat == 'sim_concat':
                self.direct_scale = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels,
                                              kernel_size=kernel_size, padding=kernel_size // 2)
            else:
                raise NotImplementedError
            constant_init(self.direct_scale, val=0.)

        out_channels_hr = 2 * groups
        if self.direction_feat == 'sim':
            self.hr_offset = nn.Conv2d(local_window ** 2 - 1, out_channels_hr,
                                       kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == 'sim_concat':
            self.hr_offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels_hr,
                                       kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.hr_direct_scale = nn.Conv2d(in_channels, out_channels_hr,
                                                 kernel_size=kernel_size, padding=kernel_size // 2)
            elif self.direction_feat == 'sim_concat':
                self.hr_direct_scale = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels_hr,
                                                 kernel_size=kernel_size, padding=kernel_size // 2)
            else:
                raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset, scale=None):
        if scale is None:
            scale = self.scale

        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale)
        coords = coords.view(B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        x_g = x.reshape(B * self.groups, -1, x.size(-2), x.size(-1))
        out = F.grid_sample(x_g, coords, mode='bilinear', align_corners=False, padding_mode="border")
        return out.view(B, -1, scale * H, scale * W)

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == 'sim':
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        else:
            raise NotImplementedError

        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return self.sample(feat2sample, offset)

    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'direct_scale'):
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale))
            scale = (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid()
            offset = offset * scale + self.init_pos
        else:
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)


class FAFCE(nn.Module):
    def __init__(self,
                 channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=False,  # use offset generator or not
                 feature_resample_group=4,
                 comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,  # for regularization, do not matter really
                 feature_resample_norm=True,
                 **kwargs):
        super().__init__()
        hr_channels, lr_channels = channels


        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        self.hamming_window = hamming_window
        self.comp_feat_upsample = comp_feat_upsample

        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        # ALPF generator
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        # AHPF generator
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)

        # optional dysampler
        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(
                in_channels=compressed_channels,
                scale=2,
                style='lp',
                groups=feature_resample_group,
                use_direct_scale=True,
                kernel_size=encoder_kernel,
                norm=feature_resample_norm)

        # hamming buffers
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer(
                'hamming_lowpass',
                torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad,
                                            lowpass_kernel + 2 * lowpass_pad))[None, None])
            self.register_buffer(
                'hamming_highpass',
                torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad,
                                            highpass_kernel + 2 * highpass_pad))[None, None])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):

        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)

        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)

        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)

        mask = mask * hamming
        mask = mask / mask.sum(dim=(-1, -2), keepdims=True)

        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, x):
        hr_feat, lr_feat = x
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        def _up_to_hr(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=compressed_hr_feat.shape[-2:], mode='nearest')

        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass)
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(
                        compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1)

                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)

                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = _up_to_hr(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2)
                    )
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = _up_to_hr(
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 2)
                    )
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else:
                    raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + _up_to_hr(self.content_encoder(compressed_lr_feat))
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + _up_to_hr(self.content_encoder2(compressed_lr_feat))
        else:
            compressed_x = _up_to_hr(compressed_lr_feat) + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            hr_feat = hr_feat_hf + hr_feat if self.hr_residual else hr_feat_hf

        if self.feature_resample:
            lr_feat = self.dysampler(hr_x=compressed_hr_feat, lr_x=compressed_lr_feat, feat2sample=lr_feat)

        return hr_feat + lr_feat