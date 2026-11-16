#!/usr/bin/env python3
"""Launch an upstream LingBot-VA server with an explicit checkpoint path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lingbot_checkpoint_contract import validate_lingbot_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--save-root", required=True)
    parser.add_argument('--reset-seed',type=int,help='Optional predeclared seed reset per episode for controlled diagnostics')
    args = parser.parse_args()

    repo = Path(args.lingbot_repo).resolve()
    model_path = Path(args.model_path).resolve()
    if not (repo / "wan_va" / "wan_va_server.py").is_file():
        raise FileNotFoundError(f"invalid LingBot repository: {repo}")
    validate_lingbot_checkpoint(model_path, args.config_name)
    sys.path.insert(0, str(repo / "wan_va"))

    import wan_va_server

    if args.reset_seed is not None:
        original_reset=wan_va_server.VA_Server._reset
        def seeded_reset(self,prompt=None):
            import random
            import numpy as np
            import torch
            random.seed(args.reset_seed);np.random.seed(args.reset_seed)
            torch.manual_seed(args.reset_seed);torch.cuda.manual_seed_all(args.reset_seed)
            result=original_reset(self,prompt=prompt)
            Path(self.exp_save_root,'reset_seed.json').write_text(json.dumps({'reset_seed':args.reset_seed,'scope':'episode; generator advances within the episode','predeclared':True},indent=2))
            return result
        wan_va_server.VA_Server._reset=seeded_reset

    # Record the text actually received at the server boundary, including the
    # released encoder's cleaning step. This does not modify either prompt.
    prompt_reset_base = wan_va_server.VA_Server._reset
    def attested_reset(self, prompt=None):
        result = prompt_reset_base(self, prompt=prompt)
        if isinstance(prompt, str):
            import hashlib
            receipt = {
                "schema_version": "lingbot_received_prompt.v1",
                "source": "server_episode_reset_before_cached_generation",
                "raw_prompt": prompt,
                "encoder_prompt": wan_va_server.prompt_clean(prompt),
                "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "config_name": args.config_name,
                "model_rng_reset_seed": args.reset_seed,
                "model_rng_policy": "predeclared_episode_reset" if args.reset_seed is not None else "upstream_unreset_random_stream",
                "prompt_rewritten_by_adapter": False,
            }
            Path(self.exp_save_root, "prompt_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
            with Path(args.save_root, "received_prompts.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt)+"\n")
        return result
    wan_va_server.VA_Server._reset = attested_reset

    if args.config_name not in wan_va_server.VA_CONFIGS:
        raise ValueError(f"unknown config {args.config_name!r}")
    config = wan_va_server.VA_CONFIGS[args.config_name]
    config.wan22_pretrained_model_name_or_path = str(model_path)
    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    wan_va_server.init_logger()
    wan_va_server.run(
        SimpleNamespace(
            config_name=args.config_name,
            port=args.port,
            save_root=str(Path(args.save_root).resolve()),
        )
    )


if __name__ == "__main__":
    main()
