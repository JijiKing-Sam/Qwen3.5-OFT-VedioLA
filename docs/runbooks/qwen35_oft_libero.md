# Qwen35 OFT LIBERO Runbook

This is the main arm of the formal comparison.
Do not interpret it alone; pair it with the controls in [controls_80k.md](/F:/videola-starvla-qwen35/docs/runbooks/controls_80k.md).

1. Bootstrap the target environment with `bash scripts/bootstrap/remote_train.sh`.
2. For inference-only sanity before any training, use [inference_eval.md](/F:/videola-starvla-qwen35/docs/runbooks/inference_eval.md).
3. Start with the LIBERO config:

```bash
bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
```

4. For smoke validation on a single GPU, keep `sdpa` from the YAML and run:

```bash
python scripts/smoke_qwen35_oft.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml
```

5. For remote training, override:

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
DATA_MIX=libero_spatial \
NUM_PROCESSES=8 \
bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
```

6. Before full evaluation, do a tiny-split overfit run and check that `normalized_actions` has shape `[B, 8, 7]`.
7. For the formal budgeted run, prefer the wrapper:

```bash
bash scripts/experiments/run_qwen35oft_80k_libero.sh
```

8. Formal comparison runs should use `80k` training steps across all groups.
9. Log the final result with [80k_comparison_template.md](/F:/videola-starvla-qwen35/reports/80k_comparison_template.md).
