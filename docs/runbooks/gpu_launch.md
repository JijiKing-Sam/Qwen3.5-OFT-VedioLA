# GPU Launch Runbook

This is the shortest path from "GPU approved" to "formal run started".
Use it for the first real `Qwen35OFT` training launch.

## Before The GPU Slot

- Keep local code on `origin` and push the exact branch you want to run.
- Freeze the benchmark target to `LIBERO-Spatial`.
- Keep the formal comparison budget at `80k steps` for all groups.
- Do not mix local dry-run artifacts with remote training outputs.

## Minimum Hardware Expectation

- Short smoke or tiny-overfit: `>= 1 GPU`, `>= 20 GB VRAM` recommended.
- Formal `Qwen35OFT` run: multi-GPU strongly preferred.
- If you have `8 x 40 GB+`, use `flash_attention_2`.
- If FlashAttention is unavailable, fall back to `sdpa` first and debug there.

## Step 1: Bootstrap The Training Environment

```bash
bash scripts/bootstrap/remote_train.sh
```

Notes:

- Default conda env name: `videola-qwen35-train`
- This installs train deps, LIBERO eval deps, editable repo, and optionally `flash-attn`

## Step 2: Preflight The GPU Node

```bash
ENV_NAME=videola-qwen35-train \
LIBERO_DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA \
ATTN_IMPLEMENTATION=flash_attention_2 \
bash scripts/bootstrap/remote_gpu_preflight.sh
```

What this checks:

- current git branch and commit
- `python`, `torch`, `transformers`, `accelerate`, `wandb`
- CUDA visibility and GPU count
- VRAM summary per GPU
- LIBERO data root exists
- output directory is writable
- `flash_attn` import if requested

## Step 3: Run A Short Smoke First

```bash
ENV_NAME=videola-qwen35-train \
LIBERO_DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA \
NUM_PROCESSES=1 \
ATTN_IMPLEMENTATION=sdpa \
bash scripts/experiments/run_qwen35oft_tiny_overfit.sh
```

Interpretation:

- This is a short training smoke, not a formal result.
- The default smoke profile freezes `qwen_vl_interface`, uses `per_device_batch_size=1`, and disables debug wait.
- The goal is to confirm loss starts moving, checkpoints save, and the job stays healthy.
- Outputs go under `results/tiny_overfit/<run_id>/`.
- This profile was verified on a single `RTX PRO 6000 96GB`; full fine-tuning of all `Qwen35OFT` parameters OOMed on that same card.
- For single-card continuation on that same `96GB` card, use `UNFREEZE_LAST_TEXT_LAYERS=2` with `sdpa`; that profile was validated through `200` steps and saved `steps_100`, `steps_200`, and `final_model`.

## Step 4: Start The Formal 80K Run

```bash
ENV_NAME=videola-qwen35-train \
LIBERO_DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA \
NUM_PROCESSES=8 \
ATTN_IMPLEMENTATION=flash_attention_2 \
WANDB_MODE=offline \
bash scripts/experiments/run_qwen35oft_80k_libero.sh
```

Notes:

- Default walltime wrapper is `7d`; step budget is still `80k`
- The wrapper runs preflight first
- Launch metadata is written into the run directory before training starts

## Resume

If the run directory already contains checkpoints and you want the trainer to resume from the latest one:

```bash
ENV_NAME=videola-qwen35-train \
LIBERO_DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA \
RUN_ID=qwen35oft_libero_80k \
IS_RESUME=true \
NUM_PROCESSES=8 \
ATTN_IMPLEMENTATION=flash_attention_2 \
bash scripts/experiments/run_qwen35oft_80k_libero.sh
```

If you want to initialize from a specific checkpoint path instead:

```bash
PRETRAINED_CHECKPOINT=/path/to/steps_xxx_pytorch_model.pt \
bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
```

## After Training

- Use the saved checkpoint under `results/.../checkpoints/`
- Record the run with [80k_comparison_template.md](/F:/videola-starvla-qwen35/reports/80k_comparison_template.md)
- Then run the benchmark-side evaluation flow from [inference_eval.md](/F:/videola-starvla-qwen35/docs/runbooks/inference_eval.md)

## Recommended Order

1. `remote_train.sh`
2. `remote_gpu_preflight.sh`
3. `run_qwen35oft_tiny_overfit.sh`
4. `run_qwen35oft_80k_libero.sh`
5. formal benchmark evaluation
