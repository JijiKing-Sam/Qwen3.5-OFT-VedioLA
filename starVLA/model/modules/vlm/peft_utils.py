from __future__ import annotations

import logging
from typing import Any, Iterable

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    get_peft_model = None


_FALLBACK_LOGGER = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_lora_config(vlm_cfg: Any) -> Any:
    if vlm_cfg is None:
        return None
    return vlm_cfg.get("lora", None)


def is_lora_enabled(vlm_cfg: Any) -> bool:
    lora_cfg = get_lora_config(vlm_cfg)
    if lora_cfg is None:
        return False
    return _as_bool(lora_cfg.get("enabled", False))


def apply_lora_if_configured(model, vlm_cfg, logger=None):
    lora_cfg = get_lora_config(vlm_cfg)
    if not is_lora_enabled(vlm_cfg):
        return model

    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "LoRA was enabled in the config, but `peft` is not installed. "
            "Install it via requirements/train.txt or `pip install peft`."
        )

    rank = int(lora_cfg.get("rank", 64))
    alpha = int(lora_cfg.get("alpha", 2 * rank))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get("target_modules", "all-linear")
    init_lora_weights = lora_cfg.get("init_lora_weights", "gaussian")

    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            init_lora_weights=init_lora_weights,
        ),
    )

    if logger is not None:
        try:
            logger.info(
                "Enabled LoRA for VLM backbone (rank=%s, alpha=%s, dropout=%s, target_modules=%s, init=%s)",
                rank,
                alpha,
                dropout,
                target_modules,
                init_lora_weights,
            )
        except RuntimeError:
            _FALLBACK_LOGGER.info(
                "Enabled LoRA for VLM backbone (rank=%s, alpha=%s, dropout=%s, target_modules=%s, init=%s)",
                rank,
                alpha,
                dropout,
                target_modules,
                init_lora_weights,
            )

    if hasattr(wrapped, "print_trainable_parameters"):
        wrapped.print_trainable_parameters()

    return wrapped


def is_peft_model_instance(model) -> bool:
    return PeftModel is not None and isinstance(model, PeftModel)


def _candidate_key_variants(key: str) -> Iterable[str]:
    candidates = []

    def add(candidate: str) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add(key)

    if ".model." in key and ".base_model.model." not in key:
        add(key.replace(".model.", ".model.base_model.model.", 1))

    snapshot = list(candidates)
    for candidate in snapshot:
        if candidate.endswith(".weight"):
            add(candidate[: -len(".weight")] + ".base_layer.weight")
        elif candidate.endswith(".bias"):
            add(candidate[: -len(".bias")] + ".base_layer.bias")

    return candidates


def remap_checkpoint_keys_for_peft(model, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """
    Expand a plain backbone checkpoint so it can be loaded into a PEFT-wrapped model.
    """
    if not checkpoint or not isinstance(checkpoint, dict):
        return checkpoint

    try:
        target_keys = set(model.state_dict().keys())
    except Exception:
        return checkpoint

    expanded = dict(checkpoint)
    remapped = 0

    for key, value in checkpoint.items():
        if key in target_keys:
            continue

        matched_candidate = None
        for candidate in _candidate_key_variants(key):
            if candidate != key and candidate in target_keys and candidate not in expanded:
                expanded[candidate] = value
                remapped += 1
                matched_candidate = candidate

        if matched_candidate is not None and key in expanded and key not in target_keys:
            del expanded[key]

    if remapped > 0:
        print(f"Enabled PEFT checkpoint compatibility remap for {remapped} tensors.")

    return expanded
