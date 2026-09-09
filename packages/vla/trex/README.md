# T-Rex

A ROCm-enabled container for [**T-Rex**](https://github.com/ZhuoyangLiu2005/T-Rex),
a tactile-reactive vision-language-action model for dexterous manipulation.

## Build & Run

```sh
ryzers build trex
ryzers run
```

`ryzers run` benchmarks the two inference stages: a slow stage that encodes the
cameras and language, and a fast stage that refines the action chunk from fresh
tactile input without re-running the visual tower.

**NOTE**: the first run downloads an 8.5 GB checkpoint from Hugging Face into
`~/.cache/huggingface`. It is public and needs no token.

## Policy server

```sh
ryzers run /ryzers/demo_trex.py
```

Serves T-Rex's ZMQ protocol on port 5678 for a robot client. Requests are
pickled dicts with `mode`, three PNG camera images, `tactile_f6` (10, 6) and
`tactile_deform` (10, 240, 240); responses carry a (16, 62) action chunk. The
robot side (a bimanual Dexmate Vega-1 with Sharpa Wave hands) was not available
here, so only the compute path is verified.

## References

- T-Rex: <https://github.com/ZhuoyangLiu2005/T-Rex>
- Paper: <https://arxiv.org/abs/2606.17055>

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.