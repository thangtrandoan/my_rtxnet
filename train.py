import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import LLVIPEnhancementDataset
from evaluate import evaluate
from model import build_model
from utils import load_checkpoint, save_checkpoint, set_seed


def get_position_from_periods(iteration: int, cumulative_period: list[int]) -> int:
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


class CosineAnnealingRestartCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    """BasicSR scheduler used by the official RT-X Net config."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        periods: list[int],
        restart_weights: list[float],
        eta_mins: list[float],
        last_epoch: int = -1,
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cumulative_period = [sum(periods[: i + 1]) for i in range(len(periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]
        return [
            eta_min
            + current_weight
            * 0.5
            * (base_lr - eta_min)
            * (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


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


def apply_mixup(batch: dict, beta: float, use_identity: bool) -> dict:
    if beta <= 0:
        return batch

    low_rgb = batch["low_rgb"]
    batch_size = low_rgb.shape[0]
    if batch_size < 2:
        return batch
    if use_identity and torch.rand(()) < 0.5:
        return batch

    lam = torch.distributions.Beta(beta, beta).sample().item()
    index = torch.randperm(batch_size)
    for key in ("low_rgb", "infrared", "gt_rgb"):
        batch[key] = batch[key] * lam + batch[key][index] * (1.0 - lam)
    return batch


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)


def build_scheduler(args, optimizer: torch.optim.Optimizer) -> CosineAnnealingRestartCyclicLR:
    return CosineAnnealingRestartCyclicLR(
        optimizer,
        periods=args.scheduler_periods,
        restart_weights=args.scheduler_restart_weights,
        eta_mins=args.scheduler_eta_mins,
    )


def align_scheduler_lr(optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingRestartCyclicLR, global_step: int) -> None:
    if global_step <= 0:
        return
    scheduler.last_epoch = global_step
    for group, lr in zip(optimizer.param_groups, scheduler.get_lr()):
        group["lr"] = lr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../LLVIP")
    parser.add_argument("--save-dir", default="runs/default")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[1, 2, 2])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-periods", type=int, nargs="+", default=[46000, 104000])
    parser.add_argument("--scheduler-restart-weights", type=float, nargs="+", default=[1.0, 1.0])
    parser.add_argument("--scheduler-eta-mins", type=float, nargs="+", default=[3e-4, 1e-6])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-every-iters", type=int, default=1000)
    parser.add_argument("--save-every-iters", type=int, default=1000)
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixup-beta", type=float, default=1.2)
    parser.add_argument("--mixup-use-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)

    train_loader = build_loader(args, "train", True)
    val_loader = build_loader(args, "test", False)

    model = build_model(channels=args.channels, stage=args.stage, num_blocks=args.num_blocks).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.L1Loss()

    start_epoch = 1
    global_step = 0
    best_psnr = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_psnr = float(checkpoint.get("best_psnr", -1.0))
        align_scheduler_lr(optimizer, scheduler, global_step)

    if args.epochs is None:
        remaining_iters = max(args.total_iters - global_step, 0)
        args.epochs = max(1, math.ceil(remaining_iters / max(len(train_loader), 1)))

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(
        "Train config: "
        f"total_iters={args.total_iters} batch={args.batch_size} patch={args.patch_size} "
        f"channels={args.channels} stage={args.stage} num_blocks={args.num_blocks} "
        f"optimizer=adam lr={args.lr:g} scheduler=official_cyclic mixup={args.mixup}"
    )
    print(f"Save dir: {save_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            if global_step >= args.total_iters:
                break

            if args.mixup:
                batch = apply_mixup(batch, args.mixup_beta, args.mixup_use_identity)
            low_rgb = batch["low_rgb"].to(device)
            infrared = batch["infrared"].to(device)
            gt_rgb = batch["gt_rgb"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(low_rgb, infrared)
            loss = criterion(pred, gt_rgb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            if batch_idx % 20 == 0 or batch_idx == len(train_loader) or global_step >= args.total_iters:
                avg = running_loss / batch_idx
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch:03d} iter {batch_idx:04d}/{len(train_loader)} "
                    f"step {global_step:06d}/{args.total_iters} loss {avg:.5f} lr {lr:.6g}"
                )

            if args.val_every_iters > 0 and global_step % args.val_every_iters == 0:
                scores = evaluate(model, val_loader, device)
                if scores["psnr"] > best_psnr:
                    best_psnr = scores["psnr"]
                    save_checkpoint(save_dir / "best.pth", model, optimizer, epoch, global_step, best_psnr)
                print(
                    f"step {global_step:06d} val PSNR {scores['psnr']:.4f} "
                    f"SSIM {scores['ssim']:.4f} best {best_psnr:.4f}"
                )
                append_csv(
                    save_dir / "log.csv",
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": running_loss / max(batch_idx, 1),
                        "lr": optimizer.param_groups[0]["lr"],
                        **scores,
                    },
                )

            if args.save_every_iters > 0 and global_step % args.save_every_iters == 0:
                save_checkpoint(save_dir / "last.pth", model, optimizer, epoch, global_step, best_psnr)

        avg_loss = running_loss / max(len(train_loader), 1)
        row = {"epoch": epoch, "step": global_step, "loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
        append_csv(save_dir / "log.csv", row)

        if global_step >= args.total_iters:
            break

    save_checkpoint(save_dir / "last.pth", model, optimizer, epoch, global_step, best_psnr)


if __name__ == "__main__":
    main()
