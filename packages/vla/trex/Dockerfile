# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Pinned to tested T-Rex commit
ARG TREX_REF=f88e10c61da123c68bf0927cf4860bc97a0381f3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TREX_ROOT=/ryzers/T-Rex \
    PYTHONPATH=/ryzers/T-Rex \
    HF_HUB_DISABLE_XET=1

WORKDIR /ryzers

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Protect the base image's ROCm torch from being replaced by a CUDA build.
RUN python3 - <<'PY' > /ryzers/constraints.txt
import torch, torchvision
print(f"torch=={torch.__version__.split('+')[0]}")
print(f"torchvision=={torchvision.__version__.split('+')[0]}")
PY

# T-Rex's pyproject is a dependency-only manifest: its modules load via
# PYTHONPATH and are never installed as a package. Installing it with pip -e
# would enforce requires-python == "3.10.*", and uv would additionally honor
# [tool.uv.sources] and pull torch from the CUDA-12.4 index. So we install the
# dependency list by hand, omitting torch/torchvision.
# deepspeed is omitted: nothing in the repo imports it, it is only reachable
# through accelerate's multi-GPU launcher.
RUN pip3 install --no-cache-dir --break-system-packages -c /ryzers/constraints.txt \
    "transformers==4.57.3" \
    "tokenizers==0.22.2" \
    "accelerate==1.8.1" \
    "datasets==4.0.0" \
    "huggingface-hub==0.36.2" \
    "safetensors==0.5.3" \
    "pyarrow==20.0.0" \
    "numpy==2.2.6" \
    "h5py==3.16.0" \
    "pillow==12.1.1" \
    "opencv-python-headless==4.12.0.88" \
    "matplotlib==3.10.3" \
    "timm==1.0.17" \
    "pyzmq==27.1.0" \
    "wandb==0.18.3" \
    "tqdm==4.67.1" \
    "av>=13.0"

# FIX: torchcodec's PyPI wheel links libtorch_cuda.so / libcudart.so and can
# never load against a ROCm torch. PyAV (installed above) bundles its own
# FFmpeg and is what the repo's own dataset quickstart uses.
RUN python3 -c "import importlib.util,sys; sys.exit(1 if importlib.util.find_spec('torchcodec') else 0)" \
 || (echo 'ERROR: torchcodec present; it cannot load against ROCm torch' && exit 1)

RUN git clone https://github.com/ZhuoyangLiu2005/T-Rex.git ${TREX_ROOT} \
 && git -C ${TREX_ROOT} checkout ${TREX_REF}

RUN python3 -c "import torch; assert torch.version.hip, 'not a ROCm torch: ' + torch.__version__; print('ROCm torch OK:', torch.__version__)" \
 && pip3 check

COPY test.py /ryzers/test_trex.py
COPY demo.py /ryzers/demo_trex.py
RUN chmod +x /ryzers/test_trex.py /ryzers/demo_trex.py

# T-Rex's ZMQ policy server port, for real-robot deployment.
EXPOSE 5678

CMD /ryzers/test_trex.py
