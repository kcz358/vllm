
export VLLM_COMMIT=8e4b351a0c9e414b0c56c32cbdef51a21d1ea1be # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
# python3 -m pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 python3 -m pip install --editable .

# python3 -m pip install transformers@git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
python3 -m pip install transformers@git+https://github.com/kcz358/transformers@vllm/stable
python3 -m pip install hf_transfer
python3 -m pip install decord librosa