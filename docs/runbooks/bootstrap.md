# Bootstrap Runbook

This workspace uses two environment entry points:

- Local WSL smoke: `bash scripts/bootstrap/wsl_smoke.sh`
- Remote Linux train/eval: `bash scripts/bootstrap/remote_train.sh`

Notes:

- Both scripts install `transformers` from the current `main` branch because Qwen3.5 support may land there earlier than a packaged release.
- `wsl_smoke.sh` is intentionally FlashAttention-free; it uses `sdpa` by default.
- `remote_train.sh` installs `flash-attn` only when `INSTALL_FLASH_ATTN=1`.
- Both scripts leave torch wheel selection overrideable through `TORCH_INDEX_URL`.
- On the local `RTX 3070 8GB` smoke machine, `Qwen35OFT` forward and checkpoint roundtrip are validated, but backward still OOMs even at `train_batch_size=1`. Treat this host as inference/checkpoint-only and move training runs to Linux remote GPUs.
