import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import LLVIPEnhancementDataset
from metrics import psnr, ssim
from model import build_model
from utils import crop_to_shape, load_checkpoint, pad_to_multiple, save_image


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, save_dir: str | Path | None = None) -> dict:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total = 0
    save_dir = Path(save_dir) if save_dir else None

    for batch in loader:
        low_rgb = batch["low_rgb"].to(device)
        infrared = batch["infrared"].to(device)
        gt_rgb = batch["gt_rgb"].to(device)

        low_rgb, shape = pad_to_multiple(low_rgb, 4)
        infrared, _ = pad_to_multiple(infrared, 4)
        pred = crop_to_shape(model(low_rgb, infrared), shape)

        total_psnr += psnr(pred, gt_rgb)
        total_ssim += ssim(pred, gt_rgb)
        total += 1

        if save_dir is not None:
            for i, name in enumerate(batch["name"]):
                save_image(pred[i], save_dir / name)

    return {"psnr": total_psnr / max(total, 1), "ssim": total_ssim / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../LLVIP")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[1, 2, 2])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = LLVIPEnhancementDataset(
        root=args.data_root,
        split=args.split,
        patch_size=None,
        augment=False,
        max_samples=args.max_samples,
    )
    print("Protocol: paper-style LLVIP processed input/thermal/target folders.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(channels=args.channels, stage=args.stage, num_blocks=args.num_blocks).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    scores = evaluate(model, loader, device, args.save_dir)
    print(f"PSNR: {scores['psnr']:.4f} dB | SSIM: {scores['ssim']:.4f}")


if __name__ == "__main__":
    main()
