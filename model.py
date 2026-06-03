import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"invalid mode {mode}")

    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class IlluminationEstimator(nn.Module):
    """Official RT-X Net illumination estimator."""

    def __init__(self, n_fea_middle: int, n_fea_in: int = 4, n_fea_out: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_c = img.mean(dim=1, keepdim=True)
        x = torch.cat([img, mean_c], dim=1)
        illu_fea = self.depth_conv(self.conv1(x))
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IGMSA(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    @staticmethod
    def _heads(x: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, hd = x.shape
        d = hd // heads
        return x.view(b, n, heads, d).permute(0, 2, 1, 3)

    def forward(self, x_in: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans.flatten(1, 2)

        q = self._heads(q_inp, self.num_heads)
        k = self._heads(k_inp, self.num_heads)
        v = self._heads(v_inp, self.num_heads)
        illu_attn = self._heads(illu_attn, self.num_heads)

        v = v * illu_attn
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        IGMSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        illu = illu_fea.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 3, dim: int = 31, level: int = 2, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 4, 4]
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                    ]
                )
            )
            dim_level //= 2

        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        fea = self.embedding(x)
        fea_encoder = []
        illu_fea_list = []

        for igab, fea_downsample, illu_downsample in self.encoder_layers:
            fea = igab(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = fea_downsample(fea)
            illu_fea = illu_downsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (fea_upsample, fusion, block) in enumerate(self.decoder_layers):
            fea = fea_upsample(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = block(fea, illu_fea)

        return self.mapping(fea) + x


def pca_reduce_channels(features: torch.Tensor, channels: int) -> torch.Tensor:
    """Reduce concatenated RGB/thermal features as in the official RT-X Net.

    The official code uses sklearn PCA on detached CPU tensors and then moves
    the result with `.cuda()`. This PyTorch version preserves that detached PCA
    behavior while remaining device agnostic.
    """
    b, c, h, w = features.shape
    if c == channels:
        return features

    x = features.detach().permute(0, 2, 3, 1).reshape(-1, c)
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    components = vh[:channels].to(dtype=features.dtype, device=features.device)
    reduced = x.to(dtype=features.dtype, device=features.device) @ components.t()
    return reduced.view(b, h, w, channels).permute(0, 3, 1, 2).contiguous()


class RTxNetSingleStage(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 40, level: int = 2, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]
        self.n_feat = n_feat
        self.estimator_rgb = IlluminationEstimator(n_feat)
        self.estimator_thermal = IlluminationEstimator(n_feat)
        self.cross_attention = IGAB(dim=n_feat, dim_head=n_feat // 4, heads=4, num_blocks=1)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img_rgb, img_thermal = inputs
        illu_fea_rgb, illu_map_rgb = self.estimator_rgb(img_rgb)
        illu_fea_thermal, _ = self.estimator_thermal(img_thermal)

        enhanced_fea_rgb = self.cross_attention(illu_fea_rgb, illu_fea_rgb)
        enhanced_fea_thermal = self.cross_attention(illu_fea_thermal, illu_fea_thermal)
        merged_features = torch.cat([enhanced_fea_rgb, enhanced_fea_thermal], dim=1)
        reduced_features = pca_reduce_channels(merged_features, self.n_feat)

        input_img = img_rgb * illu_map_rgb + img_rgb
        return self.denoiser(input_img, reduced_features)


class RTxNet(nn.Module):
    """Official RT-X Net architecture with a simple two-input forward."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 40,
        stage: int = 1,
        num_blocks=None,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]
        self.body = nn.ModuleList(
            [
                RTxNetSingleStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_feat=n_feat,
                    level=2,
                    num_blocks=num_blocks,
                )
                for _ in range(stage)
            ]
        )

    def forward(self, low_rgb: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        out = low_rgb
        for stage in self.body:
            out = stage((out, infrared))
        return out


def build_model(channels: int = 40, stage: int = 1, num_blocks=None) -> RTxNet:
    if num_blocks is None:
        num_blocks = [1, 2, 2]
    return RTxNet(n_feat=channels, stage=stage, num_blocks=num_blocks)
