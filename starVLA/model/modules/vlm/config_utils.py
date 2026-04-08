from __future__ import annotations

import inspect
from typing import Any, Tuple


def get_active_vlm_config(config: Any) -> Tuple[str, Any]:
    """
    Resolve the active VLM namespace from framework config.

    Priority:
      1. `framework.qwen35`
      2. `framework.qwenvl`
    """
    framework_cfg = getattr(config, "framework", None)
    if framework_cfg is None:
        raise AttributeError("config.framework is required to resolve the VLM backend")

    for namespace in ("qwen35", "qwenvl"):
        namespace_cfg = framework_cfg.get(namespace, None)
        if namespace_cfg is not None and namespace_cfg.get("base_vlm", None):
            return namespace, namespace_cfg

    raise KeyError("Expected one of config.framework.qwen35.base_vlm or config.framework.qwenvl.base_vlm")


def get_active_vlm_model_id(config: Any) -> str:
    _, namespace_cfg = get_active_vlm_config(config)
    return namespace_cfg.get("base_vlm")


def get_active_vlm_attn_implementation(config: Any, default: str = "sdpa") -> str:
    _, namespace_cfg = get_active_vlm_config(config)
    return namespace_cfg.get("attn_implementation", default)


def maybe_disable_thinking(model: Any = None, processor: Any = None) -> None:
    """
    Best-effort switch-off for Qwen3.5 default thinking mode.
    """
    candidates = []
    if model is not None:
        candidates.append(getattr(model, "generation_config", None))
        candidates.append(getattr(model, "config", None))
    if processor is not None:
        candidates.append(processor)
        candidates.append(getattr(processor, "tokenizer", None))

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "enable_thinking"):
            setattr(candidate, "enable_thinking", False)


def get_chat_template_kwargs(processor: Any) -> dict:
    """
    Build chat-template kwargs and disable thinking when the processor supports it.
    """
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }

    try:
        signature = inspect.signature(processor.apply_chat_template)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "processor_kwargs" in signature.parameters:
        kwargs["processor_kwargs"] = {"padding": True}
    else:
        kwargs["padding"] = True

    if signature is not None and "enable_thinking" in signature.parameters:
        kwargs["enable_thinking"] = False

    return kwargs
