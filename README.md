# my_rtxnet

Simplified-file RT-X Net implementation for RGB-thermal low-light enhancement.

The code keeps the official RT-X Net network design from `rt-xnet`
(`Illumination_Estimator`, `IGAB`, `Denoiser`, `RTxNet`) but avoids the BasicSR
project structure. The main files are `model.py`, `dataset.py`, `train.py`,
`evaluate.py`, `evaluate_vtiee.py`, and `infer.py`.

Install the required packages:

```powershell
pip install torch pillow numpy scikit-learn matplotlib lpips datasets
```

## Pipeline

This version only supports the paper-style processed layout. Raw LLVIP
`visible/infrared` evaluation and synthetic low-light generation were removed
because they produce inflated scores that are not comparable with the paper.

Expected LLVIP layout:

```text
root/
  train/input
  train/thermal
  train/target
  test/input
  test/thermal
  test/target
```

If you only have raw LLVIP `visible/infrared`, create the processed layout first:

```powershell
python prepare_llvip_synthetic.py --source-root ..\LLVIP --output-root ..\LLVIP_processed
```

This creates `input` by darkening `visible` and adding read/shot-like noise,
then copies `visible` to `target` and `infrared` to `thermal`.

Training objective:

```text
model(input_rgb, thermal) -> enhanced_rgb
L1(enhanced_rgb, target_rgb)
```

## Train

```powershell
cd D:\projects\project\my_rtxnet
python train.py --data-root path\to\processed_LLVIP --save-dir runs\paper_like --total-iters 20000 --batch-size 8 --patch-size 128 --channels 40 --stage 1 --num-blocks 1 2 2 --lr 2e-4 --mixup --mixup-beta 1.2 --grad-clip 0.01 --seed 100 --device cuda
```

This mirrors the official `Options/RTxNet_LLVIP.yml` core settings: RTxNet,
`n_feat=40`, `stage=1`, `num_blocks=[1,2,2]`, 128x128 patches, batch size 8,
Adam at `2e-4`, L1 loss, gradient clipping at `0.01`, mixup beta `1.2`, 20k
iterations, validation/checkpointing every 1000 iterations, and the BasicSR
`CosineAnnealingRestartCyclicLR` schedule.

CPU smoke test:

```powershell
python train.py --data-root path\to\processed_LLVIP --total-iters 2 --batch-size 1 --channels 8 --stage 1 --num-blocks 1 1 1 --patch-size 32 --max-train-samples 2 --max-val-samples 1 --num-workers 0 --no-mixup --val-every-iters 0 --save-every-iters 0 --device cpu
```

Checkpoints and logs are saved to `runs/default` by default.

Training also writes:

```text
log.csv
loss_curve.png
eval_curves.png
```

## Evaluate

```powershell
python evaluate.py --data-root path\to\processed_LLVIP --checkpoint runs\paper_like\best.pth --channels 40 --stage 1 --num-blocks 1 2 2 --save-dir runs\paper_like\llvip_results
```

## Evaluate V-TIEE

```powershell
python evaluate_vtiee.py --data-root ..\V-TIEE --checkpoint runs\paper_like\best.pth --channels 40 --stage 1 --num-blocks 1 2 2 --input-gain max --input-exposure min --target-gain min --target-exposure max --save-dir runs\paper_like\vtiee_results
```

V-TIEE reports `LPIPS` and `SSIM`, matching the paper table.

## Infer

Use `--input` for a low-light RGB image and `--infrared` for its aligned LLVIP infrared pair.

```powershell
python infer.py --input path\to\low_rgb.jpg --infrared path\to\infrared.jpg --checkpoint runs\default\best.pth --output output.jpg
```
