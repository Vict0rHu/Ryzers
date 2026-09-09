#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Smoke test and latency benchmark for T-Rex on Ryzen AI.

T-Rex splits inference into a slow tick (vision + language + the first
6 of 10 flow-matching steps) and a fast tick (tactile expert, remaining
4 steps) so touch can steer an action chunk without re-running the visual
tower. This measures both, since the fast tick is what bounds the control
rate on a real robot.

Observations are synthetic tensors of the correct shapes: latency depends
on architecture and hardware, not on pixel values. The repo's other VLA
ryzers do the same (smolvla uses random tensors, openpi uses zeros).
"""

import importlib.util
import io
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.environ.get("TREX_ROOT", "/ryzers/T-Rex")
CKPT_REPO = os.environ.get(
    "TREX_CKPT_REPO", "miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6")
CKPT_REV = os.environ.get(
    "TREX_CKPT_REV", "62efb3bcb45a3df0e088c8909d759b582cfb98af")

N_WARMUP = max(1, int(os.environ.get("TREX_WARMUP", 1)))
N_SLOW = int(os.environ.get("TREX_ITERS_SLOW", 3))
N_FAST = int(os.environ.get("TREX_ITERS_FAST", 10))

# T-Rex's shipped robot config (hardware_code/config/default.yaml: chunk 16,
# all 16 executed, refinement at offsets 4/8/12, 30 Hz control) implies one
# slow tick and four fast ticks per 533 ms chunk.
TARGET_SLOW_MS = 16 * (1000.0 / 30.0)
TARGET_FAST_MS = 4 * (1000.0 / 30.0)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def load_server_module():
    sys.path.insert(0, PROJECT_ROOT)
    path = os.path.join(PROJECT_ROOT, "scripts", "test.py")
    spec = importlib.util.spec_from_file_location("trex_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_args(ckpt_dir):
    # Values that model_load() does NOT auto-detect and whose defaults
    # (31 / 8 / 0) are wrong for this checkpoint.
    return types.SimpleNamespace(
        checkpoint_path=ckpt_dir,
        base_model_path="",
        stats_path="",
        dataset_name="",          # falls back to the first key in stats_data.json
        action_dim=62,
        action_chunk=16,
        use_robot_state=0,
        use_tactile_deform=1,
        use_tactile_vec=1,
        image_size=[384, 288],
        cuda="0",
        port=5678,
        disable_tactile=0,
        # Auto-detected from training_args.json when left at these defaults.
        tactile_intermediate_size=0,
        n_flare_tokens_per_frame=0,
        n_flare_steps=0,
        cascaded_total_steps=10,
        cascaded_split_step=6,
        use_tactile_code=0,
        vqvae_codebook_size=64,
        vqvae_ckpt="",
        vqvae_config=None,
    )


def png_bytes(w, h):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_payload():
    rng = np.random.default_rng(0)
    return {
        "task_description": "pick up the object",
        "image_head": png_bytes(640, 360),
        "image_wrist_right": png_bytes(640, 360),
        "image_wrist_left": png_bytes(640, 360),
        # 10 fingertips x 6-axis wrench
        "tactile_f6": rng.normal(size=(10, 6)).astype(np.float32),
        # 10 fingertips x 240x240 deformation depth map
        "tactile_deform": rng.random((10, 240, 240)).astype(np.float32),
    }


def timed(fn, n):
    lat = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return out, lat


def stats(lat):
    a = np.array(lat)
    return a.mean(), a.std(), a.min(), a.max()


def verdict(mean_ms, target_ms):
    hz = 1000.0 / mean_ms
    ok = "PASS" if mean_ms <= target_ms else "SHORT"
    return f"{ok} ({hz:.2f} Hz, target {1000.0 / target_ms:.2f} Hz, {mean_ms / target_ms:.2f}x budget)"


def main():
    banner("T-Rex on Ryzen AI")
    assert torch.version.hip, f"not a ROCm torch build: {torch.__version__}"
    print(f"torch      : {torch.__version__}")
    print(f"hip        : {torch.version.hip}")
    assert torch.cuda.is_available(), "no ROCm device visible; check /dev/kfd and /dev/dri"
    print(f"device     : {torch.cuda.get_device_name(0)}")
    print(f"arch       : {torch.cuda.get_device_properties(0).gcnArchName}")

    from huggingface_hub import snapshot_download
    print(f"\nFetching checkpoint {CKPT_REPO}@{CKPT_REV[:8]} (8.5 GB on first run)...")
    ckpt_dir = snapshot_download(repo_id=CKPT_REPO, revision=CKPT_REV)

    srv = load_server_module()
    args = build_args(ckpt_dir)

    banner("Loading model")
    t0 = time.perf_counter()
    model, processor, statistic = srv.model_load(args)
    if os.environ.get("TREX_COMPILE", "0") == "1":
        model.model = torch.compile(model.model, dynamic=True)
        print("torch.compile applied to model.model (dynamic=True)")
    print(f"[timing] model_load: {time.perf_counter() - t0:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters : {n_params:,}")

    server = srv.CascadedServer(args, model, processor, statistic)
    payload = make_payload()

    banner("Warm-up")
    for _ in range(N_WARMUP):
        r = server.predict("slow_and_fast", payload)
    print(f"action shape: {np.array(r['actions']).shape}")
    assert np.array(r["actions"]).shape == (args.action_chunk, args.action_dim)
    assert np.isfinite(np.array(r["actions"])).all(), "non-finite action"

    torch.cuda.reset_peak_memory_stats()

    banner(f"slow_and_fast  ({N_SLOW} iterations)")
    _, slow_lat = timed(lambda: server.predict("slow_and_fast", payload), N_SLOW)
    m, s, lo, hi = stats(slow_lat)
    print(f"mean {m:.1f} ms  std {s:.1f}  min {lo:.1f}  max {hi:.1f}")
    print(f"  vs robot config : {verdict(m, TARGET_SLOW_MS)}")

    banner(f"fast tick  ({N_FAST} iterations)")
    _, fast_lat = timed(lambda: server.predict("fast", payload), N_FAST)
    m, s, lo, hi = stats(fast_lat)
    print(f"mean {m:.1f} ms  std {s:.1f}  min {lo:.1f}  max {hi:.1f}")
    print(f"  vs robot config : {verdict(m, TARGET_FAST_MS)}")

    banner("Memory")
    print(f"peak allocated : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print(f"peak reserved  : {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

    print("\nTests passed!")


if __name__ == "__main__":
    main()
