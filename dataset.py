from pathlib import Path
import random

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils import list_images, pil_to_tensor


class LLVIPEnhancementDataset(Dataset):
    """Paper-style LLVIP dataset.

    Expected layout:
        root/{train,test}/input
        root/{train,test}/thermal
        root/{train,test}/target

    Raw LLVIP visible/infrared folders are intentionally not supported here:
    evaluating by synthesizing low-light from the target gives inflated scores
    and is not comparable with the RT-X Net paper.
    """

    def __init__(
        self,
        root: str | Path = "../LLVIP",
        split: str = "train",
        patch_size: int | None = 128,
        augment: bool = True,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"

        split_root = self.root / split
        self.low_dir = split_root / "input"
        self.infrared_dir = split_root / "thermal"
        self.gt_dir = split_root / "target"
        if not (self.low_dir.is_dir() and self.infrared_dir.is_dir() and self.gt_dir.is_dir()):
            raise FileNotFoundError(
                f"Expected processed LLVIP layout under {split_root}: input, thermal, target. "
                "Raw visible/infrared LLVIP is not supported for paper-style training/evaluation."
            )

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

        return {
            "low_rgb": low_rgb,
            "infrared": infrared,
            "gt_rgb": gt_rgb,
            "name": low_path.name,
            "low_path": str(low_path),
            "infrared_path": str(infrared_path),
            "gt_path": str(gt_path),
            "rgb": low_rgb,
            "thermal": infrared,
            "target": gt_rgb,
        }
