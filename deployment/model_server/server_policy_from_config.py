import argparse
import logging
import socket

import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework import build_framework


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


def main(args) -> None:
    cfg = load_config(
        config_yaml=args.config_yaml,
        framework_name=args.framework_name,
        base_vlm=args.base_vlm,
    )
    vla = build_framework(cfg)

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
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--framework_name", type=str, default=None)
    parser.add_argument("--base_vlm", type=str, default=None)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--metadata_env", type=str, default="eval_sanity")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
