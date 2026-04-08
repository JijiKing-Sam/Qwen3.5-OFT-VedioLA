# LIBERO Dry-Run

This runbook is for the new-backbone sanity path, not for reporting meaningful model quality.

## 1. Bootstrap the isolated LIBERO eval environment

```bash
bash scripts/bootstrap/wsl_libero_eval.sh
```

This creates:

- `.venv-libero`
- `.libero-config/config.yaml`

It also installs only the benchmark-side dependencies needed for `eval_libero.py`.

## 2. Start the config-backed Qwen35OFT server

Use the existing smoke environment:

```bash
source .venv-smoke/bin/activate
export HF_HOME=/mnt/f/videola-starvla-qwen35/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/f/videola-starvla-qwen35/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/f/videola-starvla-qwen35/.cache/huggingface/transformers
export TMPDIR=/mnt/f/videola-starvla-qwen35/.cache/tmp

bash examples/LIBERO/eval_files/run_qwen35oft_from_config_server.sh
```

## 3. Create identity action statistics

```bash
source .venv-smoke/bin/activate
python scripts/write_identity_dataset_statistics.py \
  --output_path results/debug/libero_identity_stats.json \
  --unnorm_key franka
```

## 4. Run a 1-trial `libero_spatial` dry-run

```bash
source .venv-libero/bin/activate
export LIBERO_CONFIG_PATH=/mnt/f/videola-starvla-qwen35/.libero-config
export LIBERO_HOME=/mnt/f/videola-starvla-qwen35/third_party/LIBERO
export PYTHONPATH=/mnt/f/videola-starvla-qwen35/third_party/LIBERO:/mnt/f/videola-starvla-qwen35:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WS_POLICY_PING_INTERVAL=0
export WS_POLICY_PING_TIMEOUT=0

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

For the exact local flow validated on this machine:

```bash
bash scripts/run_qwen35_libero_dry_run_local.sh
```

Notes:

- This validates `server -> websocket -> LIBERO wrapper -> env step`.
- Without trained `Qwen35OFT` weights, any success rate from this run is only a pipeline sanity signal.
