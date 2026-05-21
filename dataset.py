from pathlib import Path
import random

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils import list_images, pil_to_tensor


class LLVIPEnhancementDataset(Dataset):
    """LLVIP enhancement dataset.

    Raw LLVIP layout:
        LLVIP/visible/{train,test}
        LLVIP/infrared/{train,test}

    For raw LLVIP, visible is the clean RGB target. A synthetic low-light RGB
    input is generated from the transformed visible image. The paired infrared
    image is used as the thermal guidance modality.

    Processed layout is also supported:
        root/{train,test}/input
        root/{train,test}/thermal
        root/{train,test}/target
    """

    def __init__(
        self,
        root: str | Path = "../LLVIP",
        split: str = "train",
        patch_size: int | None = 128,
        augment: bool = True,
        synthetic_lowlight: bool | None = None,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.raw_llvip = False

        split_root = self.root / split
        if (split_root / "input").is_dir():
            self.low_dir = split_root / "input"
            self.infrared_dir = split_root / "thermal"
            self.gt_dir = split_root / "target"
        elif (self.root / "visible" / split).is_dir() and (self.root / "infrared" / split).is_dir():
            self.low_dir = self.root / "visible" / split
            self.infrared_dir = self.root / "infrared" / split
            self.gt_dir = self.root / "visible" / split
            self.raw_llvip = True
        else:
            raise FileNotFoundError(
                f"Unsupported dataset layout: {self.root}. Expected raw LLVIP "
                "visible/infrared folders or processed input/thermal/target folders."
            )

        self.synthetic_lowlight = self.raw_llvip if synthetic_lowlight is None else synthetic_lowlight

        low_paths = {p.name: p for p in list_images(self.low_dir)}
        infrared_paths = {p.name: p for p in list_images(self.infrared_dir)}
        gt_paths = {p.name: p for p in list_images(self.gt_dir)}
        names = sorted(set(low_paths) & set(infrared_paths) & set(gt_paths))
        if max_samples is not None:
            names = names[:max_samples]
        if not names:
            raise RuntimeError(f"No paired images found in {self.root} split={split}")

        self.samples = [(low_paths[name], infrared_paths[name], gt_paths[name]) for name in names]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _same_crop(self, images: list[Image.Image]) -> list[Image.Image]:
        if self.patch_size is None:
            return images

        w, h = images[0].size
        crop = min(self.patch_size, w, h)
        if w == crop and h == crop:
            return images

        left = random.randint(0, w - crop)
        top = random.randint(0, h - crop)
        box = (left, top, left + crop, top + crop)
        return [image.crop(box) for image in images]

    def _same_augment(self, images: list[Image.Image]) -> list[Image.Image]:
        if not self.augment:
            return images

        if random.random() < 0.5:
            images = [ImageOps.mirror(image) for image in images]
        if random.random() < 0.5:
            images = [ImageOps.flip(image) for image in images]
        rotate_k = random.randint(0, 3)
        if rotate_k:
            images = [image.rotate(90 * rotate_k, expand=True) for image in images]
        return images

    def _degrade_visible(self, gt_rgb: torch.Tensor, index: int) -> torch.Tensor:
        """Exposure reduction x5..20 plus read/shot-like Gaussian noise."""
        if self.split == "train":
            factor = random.uniform(5.0, 20.0)
            read_noise = random.uniform(0.0, 0.015)
            shot_noise = random.uniform(0.0, 0.03)
        else:
            rng = random.Random(index)
            factor = rng.uniform(8.0, 16.0)
            read_noise = 0.0
            shot_noise = 0.0

        low_rgb = gt_rgb.clamp(0, 1) / factor
        if read_noise > 0 or shot_noise > 0:
            noise_scale = read_noise + shot_noise * torch.sqrt(low_rgb.clamp_min(1e-6))
            low_rgb = low_rgb + torch.randn_like(low_rgb) * noise_scale
        return low_rgb.clamp(0, 1)

    def __getitem__(self, index: int) -> dict:
        low_path, infrared_path, gt_path = self.samples[index]

        low_img = self._load(low_path)
        infrared_img = self._load(infrared_path)
        gt_img = self._load(gt_path)

        if infrared_img.size != low_img.size:
            infrared_img = infrared_img.resize(low_img.size, Image.BICUBIC)
        if gt_img.size != low_img.size:
            gt_img = gt_img.resize(low_img.size, Image.BICUBIC)

        low_img, infrared_img, gt_img = self._same_crop([low_img, infrared_img, gt_img])
        low_img, infrared_img, gt_img = self._same_augment([low_img, infrared_img, gt_img])

        low_rgb = pil_to_tensor(low_img)
        infrared = pil_to_tensor(infrared_img)
        gt_rgb = pil_to_tensor(gt_img)
        if self.synthetic_lowlight:
            low_rgb = self._degrade_visible(gt_rgb, index)

        return {
            "low_rgb": low_rgb,
            "infrared": infrared,
            "gt_rgb": gt_rgb,
            "name": low_path.name,
            "low_path": str(low_path),
            "infrared_path": str(infrared_path),
            "gt_path": str(gt_path),
            "synthetic_lowlight": self.synthetic_lowlight,
            "rgb": low_rgb,
            "thermal": infrared,
            "target": gt_rgb,
        }

