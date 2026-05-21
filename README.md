# my_rtxnet

Minimal RTxNet-style code for LLVIP RGB-infrared image enhancement.

This is a clean PyTorch/PIL implementation. It does not depend on BasicSR, sklearn, torchvision, or the original `rt-xnet` package.

## Pipeline

Raw LLVIP has paired visible and infrared images:

```text
LLVIP/
  visible/train
  visible/test
  infrared/train
  infrared/test
  Annotations
```

The VOC XML files in `Annotations` are person detection labels and are not used for this enhancement trainer.

For raw LLVIP, the dataset returns:

```text
gt_rgb   = transformed visible image
low_rgb  = synthetic low-light degradation of gt_rgb
infrared = transformed infrared image with the same filename
```

Training objective:

```text
model(low_rgb, infrared) -> enhanced_rgb
L1(enhanced_rgb, gt_rgb)
```

The infrared image is used as the thermal guidance modality.

Synthetic low-light generation follows the simple paper-like setting: exposure reduction by a random factor in `[5, 20]`, plus read/shot-like noise during training. Crop, flip, and rotation are applied synchronously to visible and infrared before low-light degradation.

The processed layout below is also supported. In that case `input` is used directly as `low_rgb`, `thermal` as `infrared`, and `target` as `gt_rgb`.

```text
root/
  train/input
  train/thermal
  train/target
  test/input
  test/thermal
  test/target
```

## Train

```powershell
cd D:\IMG_ENHANCE\my_rtxnet
python train.py --data-root ..\LLVIP --epochs 20 --batch-size 4 --patch-size 128
```

CPU smoke test:

```powershell
python train.py --data-root ..\LLVIP --epochs 1 --batch-size 1 --channels 8 --patch-size 64 --max-train-samples 2 --max-val-samples 1
```

Checkpoints and logs are saved to `runs/default` by default.

## Evaluate

```powershell
python evaluate.py --data-root ..\LLVIP --checkpoint runs\default\best.pth --save-dir runs\default\results
```

## Infer

Use `--input` for a low-light RGB image and `--infrared` for its aligned LLVIP infrared pair.

```powershell
python infer.py --input path\to\low_rgb.jpg --infrared path\to\infrared.jpg --checkpoint runs\default\best.pth --output output.jpg
```

