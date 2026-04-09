# 2026-04-10 Single-GPU Partial-Unfreeze

## Goal

Validate a workable single-card training recipe for `Qwen35OFT` on `LIBERO-Spatial` after full fine-tuning OOMed on `RTX PRO 6000 96GB`.

## Code Changes

- move freeze application ahead of optimizer construction in:
  - `/root/autodl-tmp/videola-starvla-qwen35/starVLA/training/train_starvla.py`
  - `/root/autodl-tmp/videola-starvla-qwen35/starVLA/training/train_starvla_cotrain.py`
- make freeze parsing robust to non-plain-string config values in:
  - `/root/autodl-tmp/videola-starvla-qwen35/starVLA/training/trainer_utils/trainer_tools.py`
- prevent empty CLI `--trainer.freeze_modules ""` from overriding generated partial-unfreeze freeze lists in:
  - `/root/autodl-tmp/videola-starvla-qwen35/scripts/experiments/run_qwen35oft_tiny_overfit.sh`

## Hardware

- provider: AutoDL
- gpu: `RTX PRO 6000 Blackwell Server Edition`
- vram: `96GB`
- processes: `1`
- attention: `sdpa`

## Validated Recipe

```bash
RUN_ID=qwen35oft_tiny_last2_200_fix3 \
NUM_PROCESSES=1 \
ATTN_IMPLEMENTATION=sdpa \
PER_DEVICE_BATCH_SIZE=1 \
STEP_BUDGET=200 \
NUM_WARMUP_STEPS=20 \
SAVE_INTERVAL=100 \
EVAL_INTERVAL=50 \
LOGGING_FREQUENCY=10 \
UNFREEZE_LAST_TEXT_LAYERS=2 \
bash scripts/experiments/run_qwen35oft_tiny_overfit.sh
```

## Outcome

- freeze applied correctly
- trainable parameters dropped from `4604.889M` to `286.034M`
- optimizer state init peak stayed around `~11GB`
- run finished successfully with exit code `0`

Artifacts:

- `steps_100`: `/root/autodl-tmp/videola-starvla-qwen35/results/tiny_overfit/qwen35oft_tiny_last2_200_fix3/checkpoints/steps_100_pytorch_model.pt`
- `steps_200`: `/root/autodl-tmp/videola-starvla-qwen35/results/tiny_overfit/qwen35oft_tiny_last2_200_fix3/checkpoints/steps_200_pytorch_model.pt`
- `final_model`: `/root/autodl-tmp/videola-starvla-qwen35/results/tiny_overfit/qwen35oft_tiny_last2_200_fix3/final_model/pytorch_model.pt`
- remote log: `/root/autodl-tmp/logs/qwen35oft_tiny_last2_200_fix3.log`

Key logged points:

- step 10: `action_dit_loss=1.563569`
- step 50: `action_dit_loss=0.620257`, `mse_score=0.078373`
- step 100: `action_dit_loss=0.714475`, `mse_score=0.052690`
- step 150: `action_dit_loss=0.289154`, `mse_score=0.062458`
- step 200: `action_dit_loss=0.277846`, `mse_score=0.073612`

## Interpretation

- single-card `full fine-tune` remains out of budget on `96GB`
- single-card `last-2-text-layers + action head` is stable enough for continued smoke, checkpointing, and later benchmark-side sanity evaluation
- next single-card extension should increase step budget before increasing the number of unfrozen layers
