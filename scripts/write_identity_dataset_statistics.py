from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="franka")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--low", type=float, default=-1.0)
    parser.add_argument("--high", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    low = [args.low] * args.action_dim
    high = [args.high] * args.action_dim
    mask = [True] * args.action_dim
    dataset_statistics = {
        args.unnorm_key: {
            "action": {
                "q01": low,
                "q99": high,
                "min": low,
                "max": high,
                "mask": mask,
            },
            "num_trajectories": 0,
            "num_transitions": 0,
        }
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_statistics, f, indent=2)

    print(f"wrote_identity_stats: {output_path.as_posix()}")
    print(f"unnorm_key: {args.unnorm_key}")


if __name__ == "__main__":
    main()
