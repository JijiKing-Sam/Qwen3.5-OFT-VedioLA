from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework import build_framework


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml",
    )
    parser.add_argument("--base_vlm", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_backward", action="store_true")
    parser.add_argument("--skip_checkpoint", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=1)
    return parser.parse_args()


def build_fake_batch(chunk_len: int, batch_size: int = 2):
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1.0, 1.0, size=(chunk_len, 7)).astype(np.float32),
        "image": [image],
        "lang": "Pick up the mug and place it on the coaster.",
    }
    return [sample for _ in range(batch_size)]


def get_reference_tensor(state_dict: dict[str, torch.Tensor]):
    for name, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            return name, tensor.detach().cpu().float().clone()
    raise RuntimeError("No floating-point tensor found in model state_dict")


def load_checkpoint_state_dict(checkpoint_path: str):
    try:
        return torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_yaml)

    if args.base_vlm:
        cfg.framework.qwen35.base_vlm = args.base_vlm

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("smoke_qwen35_oft.py requires a CUDA-enabled torch install")

    model = build_framework(cfg)
    model = model.to(args.device)
    framework_name = model.__class__.__name__
    if hasattr(model.qwen_vl_interface.model, "gradient_checkpointing_enable"):
        model.qwen_vl_interface.model.gradient_checkpointing_enable()

    chunk_len = cfg.framework.action_model.past_action_window_size + 1 + cfg.framework.action_model.future_action_window_size
    train_batch = build_fake_batch(chunk_len, batch_size=args.train_batch_size)
    eval_batch = build_fake_batch(chunk_len, batch_size=args.eval_batch_size)

    model.train()
    forward_output = model(train_batch)
    action_loss_tensor = forward_output["action_loss"]

    if not args.skip_backward:
        model.zero_grad(set_to_none=True)
        action_loss_tensor.backward()
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        predict_output = model.predict_action(eval_batch)

    if not args.skip_checkpoint:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp_file:
            state_dict = model.state_dict()
            reference_name, reference_tensor = get_reference_tensor(state_dict)
            torch.save(state_dict, tmp_file.name)
            with torch.no_grad():
                state_dict[reference_name].zero_()
            reloaded_state_dict = load_checkpoint_state_dict(tmp_file.name)
            model.load_state_dict(reloaded_state_dict, strict=True)
            reloaded_tensor = model.state_dict()[reference_name].detach().cpu().float()
            max_reload_diff = (reloaded_tensor - reference_tensor).abs().max().item()
            torch.cuda.empty_cache()
    else:
        max_reload_diff = float("nan")

    action_loss = action_loss_tensor.item()
    predicted = predict_output["normalized_actions"]

    assert predicted.shape[0] == len(eval_batch)
    assert predicted.shape[1] == chunk_len

    print("config:", Path(args.config_yaml).as_posix())
    print("framework:", framework_name)
    print("action_loss:", action_loss)
    print("predicted_shape:", tuple(predicted.shape))
    print("train_batch_size:", args.train_batch_size)
    print("eval_batch_size:", args.eval_batch_size)
    print("backward_ran:", not args.skip_backward)
    print("checkpoint_roundtrip:", not args.skip_checkpoint)
    print("max_reload_diff:", max_reload_diff)


if __name__ == "__main__":
    main()
