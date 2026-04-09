# Qwen35 OFT LIBERO Runbook

This is the main arm of the formal comparison.
Do not interpret it alone; pair it with the controls in [controls_80k.md](/F:/videola-starvla-qwen35/docs/runbooks/controls_80k.md).
The GPU bring-up sequence is frozen in [gpu_launch.md](/F:/videola-starvla-qwen35/docs/runbooks/gpu_launch.md).

1. Bootstrap the target environment with `bash scripts/bootstrap/remote_train.sh`.
2. Run the node preflight with `bash scripts/bootstrap/remote_gpu_preflight.sh`.
3. For inference-only sanity before any training, use [inference_eval.md](/F:/videola-starvla-qwen35/docs/runbooks/inference_eval.md).
4. Start with the LIBERO config:

```bash
bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
```

5. For smoke validation on a single GPU, keep `sdpa` and prefer the short wrapper:

```bash
bash scripts/experiments/run_qwen35oft_tiny_overfit.sh
```

By default this smoke wrapper now uses:

- `--trainer.freeze_modules qwen_vl_interface`
- `--datasets.vla_data.per_device_batch_size 1`
- `--is_debug false`

This is the validated single-GPU sanity profile for a `96GB` card.

6. For raw model construction smoke, run:

```bash
python scripts/smoke_qwen35_oft.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml
```

7. For remote training, override:

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
DATA_MIX=libero_spatial \
NUM_PROCESSES=8 \
bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
```

8. Before full evaluation, do the short smoke and check that checkpoints save and `normalized_actions` stays `[B, 8, 7]`.
9. For the formal budgeted run, prefer the wrapper:

```bash
bash scripts/experiments/run_qwen35oft_80k_libero.sh
```

10. Formal comparison runs should use `80k` training steps across all groups.
11. Log the final result with [80k_comparison_template.md](/F:/videola-starvla-qwen35/reports/80k_comparison_template.md).
