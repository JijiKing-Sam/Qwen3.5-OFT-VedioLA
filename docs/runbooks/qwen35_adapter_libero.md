# Qwen3.5 Adapter on LIBERO (Single-GPU First Pass)

This runbook uses StarVLA infrastructure with the `Qwen35Adapter` framework.
It keeps the OFT-style continuous action chunk output while switching to an adapter-style bridge.

## Config

- Training config: `examples/LIBERO/train_files/starvla_qwen35_adapter_libero.yaml`
- Main launcher: `examples/LIBERO/train_files/run_qwen35adapter_libero_train.sh`
- Tiny smoke launcher: `scripts/experiments/run_qwen35adapter_tiny_overfit.sh`

## Tiny Overfit (recommended first)

```bash
cd /root/autodl-tmp/videola-starvla-qwen35
export WANDB_MODE=offline
export LIBERO_DATA_ROOT=/root/autodl-tmp/LEROBOT_LIBERO_DATA
export NUM_PROCESSES=1
export ATTN_IMPLEMENTATION=sdpa
bash scripts/experiments/run_qwen35adapter_tiny_overfit.sh
```

## Formal Training (80k step budget)

```bash
cd /root/autodl-tmp/videola-starvla-qwen35
export WANDB_MODE=offline
export LIBERO_DATA_ROOT=/root/autodl-tmp/LEROBOT_LIBERO_DATA
export NUM_PROCESSES=1
export MAX_TRAIN_STEPS=80000
export RUN_ID=qwen35adapter_libero_80k
bash examples/LIBERO/train_files/run_qwen35adapter_libero_train.sh
```

## Notes

- `Qwen35Adapter` is registered by `starVLA/model/framework/QwenAdapter.py`.
- The action head auto-resolves active VLM namespace (`qwen35` or `qwenvl`).
- The default adapter recipe now tracks released `VLA-Adapter` more closely: `dual-view + proprio + LoRA`.
- Tiny-overfit runs clear `FREEZE_MODULES=qwen_vl_interface` automatically when `USE_LORA=true`, because PEFT already freezes the backbone and only keeps LoRA weights trainable.
- Older non-LoRA checkpoints can now be loaded into the LoRA-enabled config through a PEFT compatibility remap during checkpoint load.
