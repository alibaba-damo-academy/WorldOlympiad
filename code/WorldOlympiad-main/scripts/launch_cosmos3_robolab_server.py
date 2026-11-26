#!/usr/bin/env python3
"""Launch the public Cosmos3 RoboLab server with local-only inference assets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-repo", required=True)
    parser.add_argument(
        "--openpi-source",
        help=(
            "Optional OpenPI checkout containing src/openpi and packages/openpi-client/src. "
            "When omitted, use the openpi-server/openpi-client packages installed in the runtime."
        ),
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--wan-vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=28000)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for the official per-request NumPy RNG schedule.",
    )
    parser.add_argument(
        "--request-seed-offset",
        type=int,
        default=0,
        help="Advance the official request RNG by this many committed windows.",
    )
    parser.add_argument('--diagnostic-fixed-seed',action='store_true',help='Paired input diagnostics only: reuse --seed for every request; formal clients reject this server')
    parser.add_argument('--prompt-format', choices=('upstream', 'json', 'plain'), default='upstream',
                        help='Preserve upstream checkpoint/default formatting, or explicitly select a comparison format')
    parser.add_argument('--capture-requests', action='store_true',
                        help='Save exact model inputs and outputs for auditing an unmodified external client')
    parser.add_argument(
        "--decode-video",
        action="store_true",
        help="Return the official decoded samples['vision'] rollout with every response.",
    )
    args = parser.parse_args()
    if args.request_seed_offset < 0:
        raise ValueError("request seed offset must be non-negative")

    repo = Path(args.cosmos_repo).resolve()
    openpi = Path(args.openpi_source).resolve() if args.openpi_source else None
    checkpoint = Path(args.checkpoint_path).resolve()
    wan_vae = Path(args.wan_vae_path).resolve()
    if not (repo / "cosmos_framework").is_dir():
        raise FileNotFoundError(f"invalid Cosmos Framework repository: {repo}")
    if openpi is not None and not (openpi / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"invalid OpenPI source checkout: {openpi}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"invalid Cosmos checkpoint: {checkpoint}")
    if not wan_vae.is_file():
        raise FileNotFoundError(wan_vae)

    os.environ.setdefault("COSMOS_TRAINING", "0")
    sys.path.insert(0, str(repo))
    if openpi is not None:
        sys.path[:0] = [
            str(openpi / "src"),
            str(openpi / "packages" / "openpi-client" / "src"),
        ]

    import cosmos_framework.inference.model as inference_model
    from cosmos_framework.scripts import action_policy_server_robolab as server
    from cosmos_framework.utils import checkpoint_db

    class _LocalWanVaeCheckpoint:
        def download(self) -> str:
            return str(wan_vae)

    original_register_checkpoints = inference_model.register_checkpoints

    def _register_checkpoints_with_local_wan_vae() -> None:
        original_register_checkpoints()
        checkpoint_db._CHECKPOINTS[
            "s3://bucket/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
        ] = _LocalWanVaeCheckpoint()

    inference_model.register_checkpoints = _register_checkpoints_with_local_wan_vae

    class LocalCheckpointRobolabPolicyService(server.RobolabPolicyService):
        def _build_setup_args(self, service_args):  # type: ignore[no-untyped-def]
            setup_args = super()._build_setup_args(service_args)
            return setup_args.model_copy(update={"guardrails": False})

    # Keep the official server protocol and service implementation.  The
    # subclass only disables unrelated media guardrail downloads; video still
    # comes from upstream RobolabPolicyService.infer()/model.decode().
    service_args = server.RobolabServerArgs(
        checkpoint_path=str(checkpoint),
        output_dir=Path(args.output_dir).resolve(),
        port=args.port,
        seed=args.seed,
        deterministic_seed=args.diagnostic_fixed_seed,
        decode_video=args.decode_video,
        # A published training recipe is not proof of a released checkpoint's
        # prompt contract. Preserve the official server's resolution by default.
        format_prompt_as_json={'upstream': None, 'json': True, 'plain': False}[args.prompt_format],
    )
    service = LocalCheckpointRobolabPolicyService(service_args)
    effective_json = service._transform.prompt_json_formatter is not None
    if args.prompt_format != 'upstream' and effective_json != (args.prompt_format == 'json'):
        raise RuntimeError('Cosmos server did not apply the requested prompt format')
    contract = {'requested_prompt_format': args.prompt_format, 'format_prompt_as_json': effective_json,
                'checkpoint_path': str(checkpoint), 'base_seed': args.seed,
                'guidance': service.cfg.guidance, 'num_steps': service.cfg.num_steps,
                'shift': service.cfg.shift, 'action_space': service.cfg.action_space,
                'action_chunk_size': service.cfg.action_chunk_size}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'serving_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(f"[robolab-policy-server] verified format_prompt_as_json={effective_json} requested={args.prompt_format}", flush=True)

    # A resumed simulator skips already committed policy windows. Recreate the
    # exact upstream mutable RNG position, then expose the actual seed/index in
    # each otherwise-official response so the client can fail closed if the
    # server schedule ever drifts.
    original_next_seed = service._next_seed
    for _ in range(args.request_seed_offset):
        original_next_seed()
    seeds_used: list[int] = []

    def _tracked_next_seed() -> int:
        value = int(original_next_seed())
        seeds_used.append(value)
        return value

    service._next_seed = _tracked_next_seed
    original_infer = service.infer
    request_index = args.request_seed_offset

    def _tracked_infer(observation):  # type: ignore[no-untyped-def]
        nonlocal request_index
        seeds_used.clear()
        result = original_infer(observation)
        if len(seeds_used) != 1 or not isinstance(result, dict):
            raise RuntimeError("official Cosmos request seed contract changed")
        response = dict(result)
        response["worldolympiad_request_seed"] = seeds_used[0]
        response["worldolympiad_request_index"] = request_index
        response["worldolympiad_diagnostic_only"] = args.diagnostic_fixed_seed
        response['worldolympiad_format_prompt_as_json'] = effective_json
        if args.capture_requests:
            import numpy as np
            from PIL import Image
            capture = Path(args.output_dir) / 'requests' / f'request_{request_index:06d}'
            capture.mkdir(parents=True, exist_ok=False)
            Image.fromarray(server._extract_observation_image(observation)).save(capture / 'conditioning.png')
            np.save(capture / 'actions_raw.npy', np.asarray(response['action']), allow_pickle=False)
            if response.get('video') is not None:
                np.save(capture / 'decoded_video.npy', np.asarray(response['video']), allow_pickle=False)
            (capture / 'request.json').write_text(json.dumps({
                'request_index': request_index, 'seed': seeds_used[0],
                'prompt': observation['prompt'], 'format_prompt_as_json': effective_json,
                'joint_position': np.asarray(observation['observation/joint_position']).tolist(),
                'gripper_position': np.asarray(observation['observation/gripper_position']).tolist(),
                'capture_only': True,
            }, indent=2)+'\n')
        request_index += 1
        return response

    service.infer = _tracked_infer
    print(
        f"[robolab-policy-server] verified request seed offset={args.request_seed_offset}",
        flush=True,
    )

    # OmniSetup defaults to full FSDP under torchrun.  The upstream ``serve``
    # helper is single-process, however, and would make every rank bind the
    # same WebSocket port while only rank zero receives inference requests.
    # Broadcast each request so every FSDP rank enters the same collectives;
    # only rank zero exposes the otherwise unchanged OpenPI protocol.
    import torch.distributed as dist

    def make_long_inference_server():
        """Return the official OpenPI server with keepalive disabled.

        Its request handler calls the synchronous video policy and therefore
        cannot service websocket pings during a long generation.  Disabling
        protocol pings prevents a healthy inference from being mistaken for a
        dead connection; the handler and wire format are inherited unchanged.
        """

        import websockets.asyncio.server

        server_cls = server._load_openpi_websocket_policy_server()

        class _LongInferenceWebsocketPolicyServer(server_cls):
            async def run(self):
                async with websockets.asyncio.server.serve(
                    self._handler,
                    self._host,
                    self._port,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                ) as websocket_server:
                    await websocket_server.serve_forever()

        return _LongInferenceWebsocketPolicyServer

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if distributed and dist.get_world_size() > 1:
        if rank == 0:
            class _DistributedPolicy:
                def infer(self, observation):  # type: ignore[no-untyped-def]
                    payload = [observation]
                    dist.broadcast_object_list(payload, src=0)
                    return service.infer(observation)

            server_cls = make_long_inference_server()
            print(
                f"[robolab-policy-server] FSDP rank 0/{dist.get_world_size()} "
                f"serving on {socket.gethostname()}:{args.port}",
                flush=True,
            )
            server_cls(
                policy=_DistributedPolicy(), host="0.0.0.0", port=args.port, metadata={}
            ).serve_forever()
        else:
            while True:
                payload = [None]
                dist.broadcast_object_list(payload, src=0)
                service.infer(payload[0])
        return

    server_cls = make_long_inference_server()
    server_cls(policy=service, host="0.0.0.0", port=args.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
