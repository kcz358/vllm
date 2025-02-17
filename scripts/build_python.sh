
export VLLM_COMMIT=6a854c7a2bb5b8a2015bbd83d94d311b991ac45d # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python3 -m pip install --editable .

# python3 -m pip install transformers@git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
python3 -m pip install transformers@git+https://github.com/huggingface/transformers
python3 -m pip install hf_transfer