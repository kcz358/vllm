
export VLLM_COMMIT=bd56c983d6fe8ff93bddd5faaf8d96e01c90fd83 # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python3 -m pip install --no-cache-dir --editable .

# python3 -m pip install transformers@git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
python3 -m pip install transformers@git+https://github.com/kcz358/transformers
python3 -m pip install hf_transfer
python3 -m pip install decord librosa