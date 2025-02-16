export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1

# vllm serve Evo-LMM/kino-7b-qwen2_5_caps_conv --trust-remote-code --uvicorn-log-level debug --tensor-parallel-size 2
python scripts/test_serving.py --num-prompts 1
