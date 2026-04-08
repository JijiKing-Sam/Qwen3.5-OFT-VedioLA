# Inference Eval Preparation

There are two distinct evaluation goals in Phase 1:

1. `Official checkpoint baseline`: use StarVLA's published checkpoint to obtain a meaningful success rate.
2. `Qwen35OFT pipeline sanity`: use the new backbone wiring to prove that `policy server -> websocket client -> benchmark wrapper` runs end-to-end before any training.

## A. Meaningful Success Rate

Use the official checkpoint first. This is the only no-train path that can produce a meaningful success rate.

- SimplerEnv baseline:

```bash
CKPT_PATH=/path/to/official/steps_xxx_pytorch_model.pt \
bash examples/SimplerEnv/eval_files/run_qwen3vl_oft_policy_server.sh
```

- Then launch the SimplerEnv side from its own environment as described in [examples/SimplerEnv/README.md](/F:/videola-starvla-qwen35/examples/SimplerEnv/README.md).

## B. Qwen35OFT Pipeline Sanity

This path validates the new backbone without requiring a trained checkpoint.

1. Bootstrap the benchmark-side environment:

```bash
bash scripts/bootstrap/wsl_libero_eval.sh
```

2. Start a config-backed server:

```bash
bash examples/LIBERO/eval_files/run_qwen35oft_from_config_server.sh
```

3. Ping it from the same starVLA environment:

```bash
python scripts/ping_policy_server.py --host 127.0.0.1 --port 10095 --num_views 2
```

4. If you want to attach a benchmark wrapper before training, create identity action stats:

```bash
python scripts/write_identity_dataset_statistics.py \
  --output_path results/debug/libero_identity_stats.json \
  --unnorm_key franka
```

5. Use those identity stats for a dry-run LIBERO evaluation:

```bash
python examples/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port 10095 \
  --args.task-suite-name libero_spatial \
  --args.max-tasks 1 \
  --args.num-trials-per-task 1 \
  --args.pretrained-path "" \
  --args.unnorm-key franka \
  --args.action-stats-path results/debug/libero_identity_stats.json \
  --args.action-chunk-size 8 \
  --args.video-out-path results/debug/libero_qwen35_sanity
```

Or use the validated one-shot local wrapper:

```bash
bash scripts/run_qwen35_libero_dry_run_local.sh
```

Notes:

- The dry-run above validates interface compatibility only. It does not measure policy quality.
- Without a trained `Qwen35OFT` checkpoint, the action head is effectively random, so success rate should be treated as a pipeline signal, not a model result.
- The dedicated LIBERO runbook is [libero_dry_run.md](/F:/videola-starvla-qwen35/docs/runbooks/libero_dry_run.md).
