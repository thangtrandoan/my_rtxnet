import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray, quality: int) -> None:
    ensure_dir(path.parent)
    array = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(array, mode="RGB")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image_pil.save(path, quality=quality, subsampling=0, optimize=True)
    else:
        image_pil.save(path)


def make_low_light(
    gt: np.ndarray,
    rng: np.random.Generator,
    exposure_factor: float,
    shot_noise: float,
    read_noise: float,
) -> np.ndarray:
    low = np.clip(gt / exposure_factor, 0.0, 1.0)
    noise_std = read_noise + shot_noise * np.sqrt(np.maximum(low, 1e-8))
    noisy = low + rng.normal(0.0, noise_std, low.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def prepare_split(args, split: str) -> int:
    visible_dir = args.source_root / "visible" / split
    thermal_dir = args.source_root / "infrared" / split
    if not visible_dir.is_dir() or not thermal_dir.is_dir():
        raise FileNotFoundError(f"Expected raw LLVIP folders: {visible_dir} and {thermal_dir}")

    out_root = args.output_root / split
    input_dir = out_root / "input"
    target_dir = out_root / "target"
    thermal_out_dir = out_root / "thermal"
    for path in (input_dir, target_dir, thermal_out_dir):
        ensure_dir(path)

    visible_paths = {p.name: p for p in list_images(visible_dir)}
    thermal_paths = {p.name: p for p in list_images(thermal_dir)}
    names = sorted(set(visible_paths) & set(thermal_paths))
    if args.max_samples is not None:
        names = names[: args.max_samples]

    split_offset = 0 if split == "train" else 10_000_000
    for index, name in enumerate(names):
        rng = np.random.default_rng(args.seed + split_offset + index)
        gt = load_rgb(visible_paths[name])
        thermal = load_rgb(thermal_paths[name])

        if split == "train":
            exposure_factor = rng.uniform(args.train_exposure_min, args.train_exposure_max)
            shot_noise = args.train_shot_noise
            read_noise = args.train_read_noise
        else:
            exposure_factor = args.test_exposure
            shot_noise = args.test_shot_noise
            read_noise = args.test_read_noise

        low = make_low_light(gt, rng, exposure_factor, shot_noise, read_noise)
        out_name = Path(name).with_suffix(f".{args.output_format}").name
        save_rgb(input_dir / out_name, low, args.jpeg_quality)
        save_rgb(target_dir / out_name, gt, args.jpeg_quality)
        save_rgb(thermal_out_dir / out_name, thermal, args.jpeg_quality)

    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("../LLVIP"))
    parser.add_argument("--output-root", type=Path, default=Path("../LLVIP_processed"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--train-exposure-min", type=float, default=8.0)
    parser.add_argument("--train-exposure-max", type=float, default=16.0)
    parser.add_argument("--test-exposure", type=float, default=12.0)
    parser.add_argument("--train-shot-noise", type=float, default=0.03)
    parser.add_argument("--train-read-noise", type=float, default=0.01)
    parser.add_argument("--test-shot-noise", type=float, default=0.0)
    parser.add_argument("--test-read-noise", type=float, default=0.0)
    parser.add_argument("--output-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    total = 0
    for split in args.splits:
        total += prepare_split(args, split)
    print(f"Wrote {total} paired samples to {args.output_root}")


if __name__ == "__main__":
    main()
