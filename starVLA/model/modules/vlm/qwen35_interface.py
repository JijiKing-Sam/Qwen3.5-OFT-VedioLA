from __future__ import annotations

import torch
import torch.nn as nn

from accelerate.logging import get_logger
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.model.modules.vlm.config_utils import (
    get_active_vlm_attn_implementation,
    get_active_vlm_config,
    get_chat_template_kwargs,
    maybe_disable_thinking,
)

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 support is unavailable in the installed transformers build. "
        "Use the repository bootstrap scripts to install the current transformers main branch."
    ) from import_error

logger = get_logger(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 248077
_ACTION_TOKEN_MAX = 248077 + 2047


class _QWen3_5_VL_Interface(nn.Module):
    """
    Qwen3.5 multimodal wrapper used by VideoLA's OFT path.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()

        namespace, vlm_cfg = get_active_vlm_config(config)
        model_id = vlm_cfg.get("base_vlm", "Qwen/Qwen3.5-4B")
        attn_implementation = get_active_vlm_attn_implementation(config, default="sdpa")
        trust_remote_code = vlm_cfg.get("trust_remote_code", False)

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self.config_namespace = namespace
        self.chat_template_kwargs = get_chat_template_kwargs(processor)
        self.model.config.use_cache = False

        # Align Qwen3.5 config shape with older wrappers.
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        maybe_disable_thinking(model=self.model, processor=self.processor)

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def generate(self, **kwargs):
        generation_kwargs = dict(kwargs)
        if "enable_thinking" not in generation_kwargs:
            generation_kwargs["enable_thinking"] = False
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(**generation_kwargs)
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(messages, **self.chat_template_kwargs)

        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= self._ACTION_TOKEN_MIN) & (seq <= self._ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; verify the action-tokenized tokenizer setup in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)
