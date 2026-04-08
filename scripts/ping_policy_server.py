from __future__ import annotations

import argparse

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--instruction", type=str, default="Pick up the mug and place it on the coaster.")
    return parser.parse_args()


def main():
    args = parse_args()
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = client.get_server_metadata()

    sample_image = np.random.randint(
        0,
        255,
        size=(args.image_size, args.image_size, 3),
        dtype=np.uint8,
    )
    examples = [{
        "image": [sample_image.copy() for _ in range(args.num_views)],
        "lang": args.instruction,
    }]

    response = client.predict_action({"examples": examples})
    client.close()

    normalized_actions = np.asarray(response["data"]["normalized_actions"])
    print("server_metadata:", server_metadata)
    print("normalized_actions_shape:", tuple(normalized_actions.shape))
    print("first_action:", normalized_actions[0, 0].tolist())


if __name__ == "__main__":
    main()
