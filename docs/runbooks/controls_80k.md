# 80K-Step Control Matrix

Goal: every meaningful result for `Qwen35OFT` must be interpreted against matched controls, otherwise failures cannot be localized.

## Required Groups

- Main: `Qwen35OFT` in this repo.
- Control A: `OpenVLA-OFT` reproduction in its own repo.
- Control B: collaborator memory baseline.
  Current default is the collaborator's naive token dropping / `MME-VLA` line unless a newer baseline is explicitly frozen.

## Shared Constraints

- First shared benchmark: `LIBERO-Spatial`.
- Use the same task subset and eval protocol across all groups before comparing success rate.
- Use the same training budget: `80k steps` for every formal run.
- Record the exact commit, config, checkpoint source, seed, GPU type/count, batch size, and start/end timestamps.
- Record actual walltime, but walltime is descriptive now, not the primary budget variable.
- Keep the comparison one-axis-at-a-time.
  For Phase 1, `Qwen35OFT` differs from controls mainly by backbone/policy path, not by extra token-drop logic.
- Do not mix local smoke results with formal `80k-step` runs.
  Local smoke is only for chain validation.

## Interpretation Rules

- `OpenVLA-OFT` works, collaborator control works, `Qwen35OFT` fails:
  likely our StarVLA Qwen3.5 integration or training recipe is wrong.
- `OpenVLA-OFT` works, both Qwen-based runs fail:
  likely Qwen3.5 backbone path, data formatting, or action alignment is wrong.
- `Qwen35OFT` works, collaborator control fails:
  likely the perception memory / token-drop branch is causing the regression.
- All three fail:
  likely environment, dataset, or evaluation protocol is wrong.

## Execution Order

If GPU is scarce, run in this order:

1. `OpenVLA-OFT` control
2. collaborator control
3. `Qwen35OFT` main run

Reason: two controls anchor the debugging space before we spend a full budget on the new policy.

## Wrappers

- Main run wrapper:
  `bash scripts/experiments/run_qwen35oft_80k_libero.sh`
- OpenVLA-OFT control wrapper:
  `bash scripts/experiments/run_openvla_oft_control_80k.sh`
- collaborator control wrapper:
  `bash scripts/experiments/run_collab_control_80k.sh`

All formal runs should be summarized with [80k_comparison_template.md](/F:/videola-starvla-qwen35/reports/80k_comparison_template.md).
