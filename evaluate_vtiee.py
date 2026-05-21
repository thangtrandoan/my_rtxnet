import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from metrics import psnr, ssim
from model import build_model
from utils import crop_to_shape, load_checkpoint, pad_to_multiple, pil_to_tensor, save_image


PATH_RE = re.compile(
    r"(?:^|/)V-TIEE/(?P<path>(?:RGB/(?P<scene>\d+)/gain_(?P<gain>\d+)/exposure_(?P<exposure>\d+)\.png)|(?:Thermal/\d+\.png))$"
)


@dataclass(frozen=True)
class RGBEntry:
    index: int
    path: str
    scene: int
    gain: int
    exposure: int


class VTIEEEnhancementDataset(Dataset):
    """V-TIEE dataset stored with Hugging Face `datasets.save_to_disk`.

    V-TIEE contains multiple RGB captures per scene under different gain and
    exposure settings plus one thermal image per scene. It does not define a
    single canonical low/target pair, so this wrapper selects captures by gain
    and exposure rules.
    """

    def __init__(
        self,
        root: str | Path = "../V-TIEE",
        split: str = "train",
        input_gain: str = "max",
        input_exposure: str = "min",
        target_gain: str = "min",
        target_exposure: str = "max",
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.input_gain = input_gain
        self.input_exposure = input_exposure
        self.target_gain = target_gain
        self.target_exposure = target_exposure

        try:
            from datasets import Image as HFImage
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "evaluate_vtiee.py needs the Hugging Face `datasets` package to read V-TIEE Arrow files. "
                "Install it with: pip install datasets"
            ) from exc

        dataset_dict = load_from_disk(str(self.root))
        if split not in dataset_dict:
            raise ValueError(f"Split {split!r} not found in {self.root}. Available splits: {list(dataset_dict.keys())}")

        self.dataset = dataset_dict[split].cast_column("image", HFImage(decode=False))
        self.paths = self._read_image_paths()
        rgb_entries, thermal_by_scene = self._index_entries()
        self.samples = self._build_samples(rgb_entries, thermal_by_scene)
        self.skipped_samples = len(rgb_entries) - len(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No V-TIEE samples could be built from {self.root} split={split}")

    def _read_image_paths(self) -> list[str]:
        paths: list[str | None] = []
        for i in range(len(self.dataset)):
            image = self.dataset[i]["image"]
            paths.append(image.get("path") if isinstance(image, dict) else None)

        if all(paths):
            return [self._normalize_path(path) for path in paths if path is not None]

        info_path = self.root / self.split / "dataset_info.json"
        if not info_path.is_file():
            raise RuntimeError("V-TIEE Arrow rows do not expose image paths and dataset_info.json is missing.")

        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        checksum_paths = list(info.get("download_checksums", {}).keys())
        if len(checksum_paths) != len(self.dataset):
            raise RuntimeError(
                "V-TIEE Arrow rows do not expose image paths and dataset_info.json length does not match Arrow rows."
            )
        return [self._normalize_path(path) for path in checksum_paths]

    @staticmethod
    def _normalize_path(path: str) -> str:
        path = path.replace("\\", "/")
        match = PATH_RE.search(path)
        if match:
            return match.group("path")
        if "RGB/" in path:
            return path[path.index("RGB/") :]
        if "Thermal/" in path:
            return path[path.index("Thermal/") :]
        raise ValueError(f"Cannot parse V-TIEE image path: {path}")

    def _index_entries(self) -> tuple[dict[int, list[RGBEntry]], dict[int, int]]:
        rgb_entries: dict[int, list[RGBEntry]] = {}
        thermal_by_scene: dict[int, int] = {}

        for index, path in enumerate(self.paths):
            parts = path.split("/")
            if parts[0] == "Thermal":
                thermal_by_scene[int(Path(parts[1]).stem)] = index
                continue
            if len(parts) != 4 or parts[0] != "RGB":
                continue
            scene = int(parts[1])
            gain = int(parts[2].removeprefix("gain_"))
            exposure = int(Path(parts[3]).stem.removeprefix("exposure_"))
            rgb_entries.setdefault(scene, []).append(RGBEntry(index, path, scene, gain, exposure))

        return rgb_entries, thermal_by_scene

    def _build_samples(self, rgb_entries: dict[int, list[RGBEntry]], thermal_by_scene: dict[int, int]) -> list[dict]:
        samples = []
        for scene in sorted(rgb_entries):
            entries = rgb_entries[scene]
            if scene not in thermal_by_scene:
                continue
            input_entry = self._select_entry(entries, self.input_gain, self.input_exposure)
            target_entry = self._select_entry(entries, self.target_gain, self.target_exposure)
            if input_entry is None or target_entry is None:
                continue
            samples.append(
                {
                    "scene": scene,
                    "low_index": input_entry.index,
                    "thermal_index": thermal_by_scene[scene],
                    "target_index": target_entry.index,
                    "name": f"scene_{scene:03d}_g{input_entry.gain}_e{input_entry.exposure}.png",
                    "low_path": input_entry.path,
                    "thermal_path": self.paths[thermal_by_scene[scene]],
                    "target_path": target_entry.path,
                }
            )
        return samples

    @staticmethod
    def _select_entry(entries: list[RGBEntry], gain_rule: str, exposure_rule: str) -> RGBEntry | None:
        gains = sorted({entry.gain for entry in entries})
        gain = VTIEEEnhancementDataset._resolve_rule(gain_rule, gains, "gain")
        if gain is None:
            return None
        candidates = [entry for entry in entries if entry.gain == gain]
        exposures = sorted({entry.exposure for entry in candidates})
        exposure = VTIEEEnhancementDataset._resolve_rule(exposure_rule, exposures, "exposure")
        if exposure is None:
            return None
        for entry in candidates:
            if entry.exposure == exposure:
                return entry
        return None

    @staticmethod
    def _resolve_rule(rule: str, values: list[int], field: str) -> int | None:
        if not values:
            raise ValueError(f"No values available for {field}")
        if rule == "min":
            return values[0]
        if rule == "max":
            return values[-1]
        value = int(rule)
        if value not in values:
            return None
        return value

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, index: int) -> Image.Image:
        image_record: Any = self.dataset[index]["image"]
        if isinstance(image_record, dict) and image_record.get("bytes") is not None:
            return Image.open(io.BytesIO(image_record["bytes"])).convert("RGB")
        if isinstance(image_record, dict) and image_record.get("path"):
            return Image.open(image_record["path"]).convert("RGB")
        raise RuntimeError(f"Cannot decode V-TIEE image at row {index}")

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        low_img = self._load_image(sample["low_index"])
        thermal_img = self._load_image(sample["thermal_index"])
        target_img = self._load_image(sample["target_index"])

        if thermal_img.size != low_img.size:
            thermal_img = thermal_img.resize(low_img.size, Image.BICUBIC)
        if target_img.size != low_img.size:
            target_img = target_img.resize(low_img.size, Image.BICUBIC)

        return {
            "low_rgb": pil_to_tensor(low_img),
            "infrared": pil_to_tensor(thermal_img),
            "gt_rgb": pil_to_tensor(target_img),
            "name": sample["name"],
            "scene": sample["scene"],
            "low_path": sample["low_path"],
            "infrared_path": sample["thermal_path"],
            "gt_path": sample["target_path"],
        }


@torch.no_grad()
def evaluate_vtiee(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str | Path | None = None,
    compute_lpips: bool = True,
) -> dict:
    model.eval()
    lpips_model = None
    if compute_lpips:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "V-TIEE evaluation uses LPIPS + SSIM. Install LPIPS with: pip install lpips "
                "or pass --no-lpips to report only SSIM/PSNR."
            ) from exc
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    total_lpips = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_images = 0
    save_dir = Path(save_dir) if save_dir else None

    for batch in loader:
        low_rgb = batch["low_rgb"].to(device)
        infrared = batch["infrared"].to(device)
        gt_rgb = batch["gt_rgb"].to(device)

        low_rgb, shape = pad_to_multiple(low_rgb, 4)
        infrared, _ = pad_to_multiple(infrared, 4)
        pred = crop_to_shape(model(low_rgb, infrared), shape).clamp(0, 1)

        for i in range(pred.shape[0]):
            total_ssim += ssim(pred[i : i + 1], gt_rgb[i : i + 1])
            total_psnr += psnr(pred[i : i + 1], gt_rgb[i : i + 1])

        if lpips_model is not None:
            pred_lpips = pred * 2.0 - 1.0
            gt_lpips = gt_rgb.clamp(0, 1) * 2.0 - 1.0
            total_lpips += lpips_model(pred_lpips, gt_lpips).sum().item()
        total_images += pred.shape[0]

        if save_dir is not None:
            for i, name in enumerate(batch["name"]):
                save_image(pred[i], save_dir / name)

    scores = {
        "ssim": total_ssim / max(total_images, 1),
        "psnr": total_psnr / max(total_images, 1),
    }
    if lpips_model is not None:
        scores["lpips"] = total_lpips / max(total_images, 1)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../V-TIEE")
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--input-gain", default="max", help="'min', 'max', or an integer gain value.")
    parser.add_argument("--input-exposure", default="min", help="'min', 'max', or an integer exposure value.")
    parser.add_argument("--target-gain", default="min", help="'min', 'max', or an integer gain value.")
    parser.add_argument("--target-exposure", default="max", help="'min', 'max', or an integer exposure value.")
    parser.add_argument("--no-lpips", action="store_true", help="Disable LPIPS and report only SSIM/PSNR.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = VTIEEEnhancementDataset(
        root=args.data_root,
        split=args.split,
        input_gain=args.input_gain,
        input_exposure=args.input_exposure,
        target_gain=args.target_gain,
        target_exposure=args.target_exposure,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(
        "V-TIEE samples: "
        f"{len(dataset)} | input gain={args.input_gain} exposure={args.input_exposure} | "
        f"target gain={args.target_gain} exposure={args.target_exposure}"
    )
    if dataset.skipped_samples:
        print(f"Skipped scenes without the requested gain/exposure setting: {dataset.skipped_samples}")

    model = build_model(channels=args.channels).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    scores = evaluate_vtiee(model, loader, device, args.save_dir, compute_lpips=not args.no_lpips)
    if "lpips" in scores:
        print(f"LPIPS: {scores['lpips']:.4f} | SSIM: {scores['ssim']:.4f} | PSNR(ref): {scores['psnr']:.4f} dB")
    else:
        print(f"SSIM: {scores['ssim']:.4f} | PSNR(ref): {scores['psnr']:.4f} dB")


if __name__ == "__main__":
    main()
