export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve kcz358/kino-7b-qwen2_5_ola \
    -tp 2 --port 30000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.83 \
    --limit-mm-per-prompt "image=1,video=1,audio=1" \
    --max-model-len 128000 --task embed
