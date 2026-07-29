#!/bin/bash

sudo apt-get update
python3.11 -m pip install --user --upgrade pip setuptools wheel
python3.11 -m pip install --user --upgrade vllm-tpu "huggingface_hub[hf_transfer]"

~/.local/bin/vllm --version

# This bootstraps the TPU host with vLLM. To start the benchmark's 8B server,
# use the canonical launcher (Llama-3.1-8B, prefix caching + /metrics on :8001):
#
#   bash scripts/launch_vllm_8b.sh
#
# which runs the equivalent of:
#   vllm serve "meta-llama/Llama-3.1-8B-Instruct" \
#       --port 8001 --enable-prefix-caching \
#       --max-model-len 131072 --tensor-parallel-size 4
