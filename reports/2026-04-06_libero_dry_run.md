# 2026-04-06 LIBERO Dry-Run

## Scope

- Machine: local WSL + RTX 3070 Laptop GPU 8GB
- Goal: validate `Qwen35OFT` inference pipeline on `libero_spatial`
- Mode: config-backed server, no trained `Qwen35OFT` checkpoint
- Run shape: `1 task x 1 trial`

## Command

```bash
bash scripts/run_qwen35_libero_dry_run_local.sh
```

## Result

- Exit code: `0`
- Server ping succeeded before rollout
- LIBERO environment initialized successfully
- Rollout completed and video artifact was written
- Output video:
  - `results/debug/libero_qwen35_sanity/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_failure.mp4`

## Interpretation

- This run confirms the end-to-end inference path:
  - `LIBERO -> websocket client -> Qwen35OFT server -> normalized actions -> env.step`
- The rollout ended as `failure`, which is expected because the current `Qwen35OFT` head is untrained.
- This is a pipeline sanity signal only, not a meaningful success-rate result.

## Follow-Up

- Move to remote GPU for training / overfit / formal success-rate evaluation
- Keep local usage focused on inference sanity and wrapper regression checks
- Optional cleanup later: suppress robosuite EGL teardown warnings after env shutdown
