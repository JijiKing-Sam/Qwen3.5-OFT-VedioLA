# SimplerEnv Baseline Runbook

Goal: reproduce the upstream StarVLA SimplerEnv server path before swapping the backbone.

This is an upstream StarVLA sanity baseline.
It is not a replacement for the formal `OpenVLA-OFT` control tracked in [controls_80k.md](/F:/videola-starvla-qwen35/docs/runbooks/controls_80k.md).

1. Download the official checkpoint snapshot for `StarVLA/Qwen3VL-OFT-Bridge-RT-1` to a local directory.
2. Point `CKPT_PATH` at the `.pt` checkpoint inside that snapshot.
3. Start the policy server:

```bash
CKPT_PATH=/path/to/steps_xxx_pytorch_model.pt \
bash examples/SimplerEnv/eval_files/run_qwen3vl_oft_policy_server.sh
```

4. In the separate `simpler_env` environment, follow the upstream evaluation flow from [examples/SimplerEnv/README.md](/F:/videola-starvla-qwen35/examples/SimplerEnv/README.md).
5. Record checkpoint hash, environment versions, number of episodes, success rate, VRAM, and throughput in `reports/phase1_template.md`.
