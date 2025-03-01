export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve Evo-LMM/kino-7b-qwen2_5_caps_conv \
    -tp 4 -pp 2 --port 30000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --limit-mm-per-prompt "image=1,video=1,audio=1" \
    --max-model-len 128000 --enforce-eager
