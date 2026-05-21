import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IlluminationEstimator(nn.Module):
    """Small Retinex-style illumination feature estimator."""

    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, channels // 4)
        self.features = nn.Sequential(
            nn.Conv2d(4, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 5, padding=2, groups=groups),
            nn.GELU(),
        )
        self.map = nn.Conv2d(channels, 3, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        feat = self.features(torch.cat([x, mean], dim=1))
        illum_map = torch.sigmoid(self.map(feat))
        return feat, illum_map


class GuidedUNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_proj = nn.Conv2d(3 + channels, channels, 3, padding=1)
        self.enc1 = ConvBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(channels * 2, channels * 2)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1)
        self.mid = ConvBlock(channels * 4, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec1 = ConvBlock(channels * 2, channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, low_rgb: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(torch.cat([low_rgb, guidance], dim=1))
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        mid = self.mid(self.down2(enc2))

        dec2 = self.up2(mid)
        if dec2.shape[-2:] != enc2.shape[-2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        if dec1.shape[-2:] != enc1.shape[-2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        residual = self.out(dec1)
        return torch.sigmoid(low_rgb + residual)


class RTxNetLite(nn.Module):
    """Compact RGB-infrared enhancement network inspired by RT-X Net."""

    def __init__(self, channels: int = 40):
        super().__init__()
        self.rgb_illum = IlluminationEstimator(channels)
        self.infrared_illum = IlluminationEstimator(channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2 + 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.restore = GuidedUNet(channels)

    def forward(self, low_rgb: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        rgb_feat, rgb_illum_map = self.rgb_illum(low_rgb)
        infrared_feat, _ = self.infrared_illum(infrared)
        rgb_retinex = (low_rgb * (1.0 + rgb_illum_map)).clamp(0, 1)
        guidance = self.fusion(torch.cat([rgb_feat, infrared_feat, infrared], dim=1))
        return self.restore(rgb_retinex, guidance)


def build_model(channels: int = 40) -> RTxNetLite:
    return RTxNetLite(channels=channels)

