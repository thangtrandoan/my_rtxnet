import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import LLVIPEnhancementDataset
from evaluate import evaluate
from model import build_model
from utils import load_checkpoint, save_checkpoint, set_seed


def build_loader(args, split: str, is_train: bool) -> DataLoader:
    dataset = LLVIPEnhancementDataset(
        root=args.data_root,
        split=split,
        patch_size=args.patch_size if is_train else None,
        augment=is_train,
        max_samples=args.max_train_samples if is_train else args.max_val_samples,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else args.val_batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../LLVIP")
    parser.add_argument("--save-dir", default="runs/default")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)

    train_loader = build_loader(args, "train", True)
    val_loader = build_loader(args, "test", False)

    model = build_model(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    criterion = nn.L1Loss()

    start_epoch = 1
    global_step = 0
    best_psnr = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_psnr = float(checkpoint.get("best_psnr", -1.0))

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Save dir: {save_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            low_rgb = batch["low_rgb"].to(device)
            infrared = batch["infrared"].to(device)
            gt_rgb = batch["gt_rgb"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(low_rgb, infrared)
            loss = criterion(pred, gt_rgb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                avg = running_loss / batch_idx
                print(f"epoch {epoch:03d} iter {batch_idx:04d}/{len(train_loader)} loss {avg:.5f}")

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        row = {"epoch": epoch, "step": global_step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}

        if epoch % args.val_every == 0:
            scores = evaluate(model, val_loader, device)
            row.update(scores)
            if scores["psnr"] > best_psnr:
                best_psnr = scores["psnr"]
                save_checkpoint(save_dir / "best.pth", model, optimizer, epoch, global_step, best_psnr)
            print(f"epoch {epoch:03d} val PSNR {scores['psnr']:.4f} SSIM {scores['ssim']:.4f} best {best_psnr:.4f}")

        if epoch % args.save_every == 0:
            save_checkpoint(save_dir / "last.pth", model, optimizer, epoch, global_step, best_psnr)
        append_csv(save_dir / "log.csv", row)


if __name__ == "__main__":
    main()

