# 2026-04-06 Local Smoke Report

## Metadata

- Date: 2026-04-06
- Workspace: `F:\videola-starvla-qwen35`
- Base commit: `2404dc3b0c4749c667438d301d5da45370fdfe36`
- Working tree: dirty (`Qwen35OFT` bootstrap in progress)
- Host: Windows + WSL Ubuntu-22.04
- GPU: `NVIDIA GeForce RTX 3070 Laptop GPU (8GB)`
- Torch / transformers: `2.11.0+cu128` / `5.6.0.dev0`

## Commands

Forward-only smoke:

```bash
python scripts/smoke_qwen35_oft.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml \
  --skip_backward \
  --skip_checkpoint \
  --eval_batch_size 1
```

Forward + checkpoint roundtrip:

```bash
python scripts/smoke_qwen35_oft.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml \
  --skip_backward
```

Backward probe:

```bash
python scripts/smoke_qwen35_oft.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml \
  --skip_checkpoint \
  --train_batch_size 1 \
  --eval_batch_size 1
```

## Results

- `Qwen35OFT` loads successfully in WSL smoke env.
- Fake-batch forward succeeds.
- `predict_action()` returns shape `(1, 8, 7)` in forward-only smoke and `(2, 8, 7)` in the checkpoint run.
- Checkpoint roundtrip succeeds with in-place reload validation and `max_reload_diff = 0.0`.
- Backward still fails on the local 8GB GPU with `torch.OutOfMemoryError` at `action_loss.backward()`, even after disabling `use_cache` and keeping `train_batch_size=1`.

## Logs

- Forward-only: `F:\videola-starvla-qwen35\.logs\qwen35_smoke_forward_only.log`
- Forward + checkpoint: `F:\videola-starvla-qwen35\.logs\qwen35_smoke_skip_backward.log`
- Backward probe: `F:\videola-starvla-qwen35\.logs\qwen35_smoke_backward.log`

## Conclusions

- This machine is suitable for:
  - environment bootstrap
  - model load verification
  - fake-batch forward / inference smoke
  - checkpoint save/load validation
- This machine is not suitable for:
  - Qwen3.5-OFT backward
  - local fine-tuning
  - local success-rate evaluation at training-scale settings

## Next Step

- Move tiny-split overfit and LIBERO / SimplerEnv formal runs to a Linux remote GPU machine with at least `24GB` VRAM, preferably `40GB+`.
