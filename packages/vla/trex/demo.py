#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Start T-Rex's ZMQ policy server.

T-Rex ships scripts/test.py as its server, but its __main__ path does not work
with the public checkpoint. Two problems:
  - the warm-up builds a (5, 6) tactile array while a bimanual checkpoint
    reshapes it to (10, 6), which raises;
  - argparse rejects use_tactile_code=1 without --vqvae_ckpt, but this
    checkpoint embeds its own VQ-VAE and does not need one.
This launcher calls the same model_load() and CascadedServer with a corrected
warm-up, so the protocol and behaviour are upstream's.

Protocol (pickled dict over ZMQ REQ/REP):
  request : mode ("slow" | "fast" | "slow_and_fast"), task_description,
            image_head / image_wrist_right / image_wrist_left (PNG bytes),
            tactile_f6 (10, 6), tactile_deform (10, 240, 240)
  response: status, mode, actions (16, 62), chunk_id, latency_ms
"""

import importlib.util
import io
import os
import pickle
import sys
import time
import traceback
import types

import numpy as np
import torch
import zmq
from PIL import Image

PROJECT_ROOT = os.environ.get("TREX_ROOT", "/ryzers/T-Rex")
CKPT_REPO = os.environ.get(
    "TREX_CKPT_REPO", "miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6")
CKPT_REV = os.environ.get(
    "TREX_CKPT_REV", "62efb3bcb45a3df0e088c8909d759b582cfb98af")
PORT = int(os.environ.get("TREX_PORT", 5678))


def load_server_module():
    sys.path.insert(0, PROJECT_ROOT)
    path = os.path.join(PROJECT_ROOT, "scripts", "test.py")
    spec = importlib.util.spec_from_file_location("trex_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_args(ckpt_dir):
    # action_dim / action_chunk / use_tactile_* are not auto-detected and the
    # upstream defaults (31 / 8 / 0) are wrong for this checkpoint.
    return types.SimpleNamespace(
        checkpoint_path=ckpt_dir, base_model_path="", stats_path="",
        dataset_name="", action_dim=62, action_chunk=16,
        use_robot_state=0, use_tactile_deform=1, use_tactile_vec=1,
        image_size=[384, 288], cuda="0", port=PORT, disable_tactile=0,
        tactile_intermediate_size=0, n_flare_tokens_per_frame=0,
        n_flare_steps=0, cascaded_total_steps=10, cascaded_split_step=6,
        use_tactile_code=0, vqvae_codebook_size=64, vqvae_ckpt="",
        vqvae_config=None,
    )


def png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color="black").save(buf, format="PNG")
    return buf.getvalue()


def main():
    assert torch.version.hip, f"not a ROCm torch build: {torch.__version__}"
    assert torch.cuda.is_available(), \
        "no ROCm device visible; check /dev/kfd and /dev/dri"
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).gcnArchName})")

    from huggingface_hub import snapshot_download
    ckpt_dir = snapshot_download(repo_id=CKPT_REPO, revision=CKPT_REV)

    srv = load_server_module()
    args = build_args(ckpt_dir)
    model, processor, statistic = srv.model_load(args)

    if os.environ.get("TREX_COMPILE", "0") == "1":
        print("Compiling (this takes a few minutes on first start)...")
        model.model = torch.compile(model.model, dynamic=True)

    server = srv.CascadedServer(args, model, processor, statistic)

    # FIX: upstream's warm-up uses (5, 6) / (5, 240, 240), which raises for a
    # bimanual checkpoint. Ten fingertips, six wrench components each.
    print("Warming up...")
    warm = {
        "task_description": "warmup",
        "image_head": png_bytes(640, 360),
        "image_wrist_right": png_bytes(640, 360),
        "image_wrist_left": png_bytes(640, 360),
        "tactile_f6": np.zeros((10, 6), dtype=np.float32),
        "tactile_deform": np.zeros((10, 240, 240), dtype=np.float32),
    }
    t0 = time.perf_counter()
    result = server.predict("slow_and_fast", warm)
    print(f"warm-up ok: action {np.array(result['actions']).shape}, "
          f"{time.perf_counter() - t0:.1f}s")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{PORT}")
    print(f"T-Rex policy server listening on port {PORT}")

    n = 0
    while True:
        try:
            payload = pickle.loads(socket.recv())
            result = server.predict(payload.get("mode", "slow_and_fast"), payload)
            socket.send(pickle.dumps(result))
            n += 1
            if n % 10 == 0:
                print(f"served {n} requests "
                      f"(last {result['latency_ms']:.0f} ms)")
        except Exception as e:
            traceback.print_exc()
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    main()