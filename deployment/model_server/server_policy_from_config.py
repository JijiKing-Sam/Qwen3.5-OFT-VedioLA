import argparse
import logging
import socket
from pathlib import Path

import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework import build_framework
from starVLA.model.modules.vlm.peft_utils import remap_checkpoint_keys_for_peft


def load_config(config_yaml: str, framework_name: str | None, base_vlm: str | None):
    cfg = OmegaConf.load(config_yaml)
    if framework_name:
        cfg.framework.name = framework_name
        cfg.framework.framework_py = framework_name

    if base_vlm:
        if getattr(cfg.framework, "qwen35", None) is not None:
            cfg.framework.qwen35.base_vlm = base_vlm
        elif getattr(cfg.framework, "qwenvl", None) is not None:
            cfg.framework.qwenvl.base_vlm = base_vlm

    return cfg


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    candidate = Path(checkpoint_path).expanduser()
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        for name in ("pytorch_model.pt", "pytorch_model.bin", "model.safetensors"):
            resolved = candidate / name
            if resolved.is_file():
                return resolved

    raise FileNotFoundError(f"Checkpoint path does not exist or is unsupported: {checkpoint_path}")


def load_state_dict_from_path(checkpoint_path: Path):
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def maybe_load_checkpoint(vla, checkpoint_path: str | None, strict_load: bool) -> Path | None:
    if not checkpoint_path:
        return None

    resolved_path = resolve_checkpoint_path(checkpoint_path)
    state_dict = load_state_dict_from_path(resolved_path)
    state_dict = remap_checkpoint_keys_for_peft(vla, state_dict)
    incompatible = vla.load_state_dict(state_dict, strict=strict_load)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    logging.info(
        "Loaded checkpoint %s (strict=%s, missing=%d, unexpected=%d)",
        resolved_path,
        strict_load,
        len(missing_keys),
        len(unexpected_keys),
    )
    if missing_keys:
        logging.info("Missing keys: %s", missing_keys)
    if unexpected_keys:
        logging.info("Unexpected keys: %s", unexpected_keys)
    return resolved_path


def main(args) -> None:
    cfg = load_config(
        config_yaml=args.config_yaml,
        framework_name=args.framework_name,
        base_vlm=args.base_vlm,
    )
    vla = build_framework(cfg)
    resolved_checkpoint = maybe_load_checkpoint(
        vla,
        checkpoint_path=args.pretrained_checkpoint,
        strict_load=args.strict_load,
    )

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating config-backed server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "env": args.metadata_env,
            "framework": cfg.framework.name,
            "config_yaml": args.config_yaml,
            "checkpoint": str(resolved_checkpoint) if resolved_checkpoint is not None else "",
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--framework_name", type=str, default=None)
    parser.add_argument("--base_vlm", type=str, default=None)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument(
        "--strict_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load checkpoint weights with strict key matching by default.",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--metadata_env", type=str, default="eval_sanity")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
