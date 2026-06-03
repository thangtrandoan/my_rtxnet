import argparse
from pathlib import Path

import torch
from PIL import Image

from model import build_model
from utils import crop_to_shape, load_checkpoint, pad_to_multiple, pil_to_tensor, save_image


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Low-light RGB image")
    parser.add_argument("--infrared", required=True, help="Aligned LLVIP infrared image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[1, 2, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(channels=args.channels, stage=args.stage, num_blocks=args.num_blocks).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    low_img = Image.open(args.input).convert("RGB")
    infrared_img = Image.open(args.infrared).convert("RGB")
    if infrared_img.size != low_img.size:
        infrared_img = infrared_img.resize(low_img.size, Image.BICUBIC)

    low_rgb = pil_to_tensor(low_img).unsqueeze(0).to(device)
    infrared = pil_to_tensor(infrared_img).unsqueeze(0).to(device)
    low_rgb, shape = pad_to_multiple(low_rgb, 4)
    infrared, _ = pad_to_multiple(infrared, 4)
    pred = crop_to_shape(model(low_rgb, infrared), shape)[0]
    save_image(pred, Path(args.output))


if __name__ == "__main__":
    main()
