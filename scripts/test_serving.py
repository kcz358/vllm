from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset

import argparse



def print_outputs(outputs, modalities):
    print(f"## {modalities}")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
    print("-" * 80)


print("=" * 80)

# In this script, we demonstrate how to pass input to the chat method:

def run_text():
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation,
                    sampling_params=sampling_params,
                    use_tqdm=False)
    print_outputs(outputs, "text")



audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?"
}


# Ultravox 0.3

# Qwen2-Audio
def run_audio(question: str, audio_count: int):
    audio_in_prompt = "".join([
        f"Audio {idx+1}: "
        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n" for idx in range(audio_count)
    ])

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              f"{audio_in_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def run_qwen2_vl(question: str):
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def get_multi_modal_input(modality):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if modality == "image":
        # Input image and question
        image = ImageAsset("cherry_blossom") \
            .pil_image.convert("RGB")
        img_question = "What is the content of this image?"

        return {
            "data": image,
            "question": img_question,
        }

    if modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)



def main(args):
    run_text()

    audio_count = args.num_audios
    llm, prompt, stop_token_ids = run_audio(
        question_per_audio_count[audio_count], audio_count)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=stop_token_ids)

    mm_data = {}
    if audio_count > 0:
        mm_data = {
            "audio": [
                asset.audio_and_sample_rate
                for asset in audio_assets[:audio_count]
            ]
        }

    assert args.num_prompts > 0
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}
    if args.num_prompts > 1:
        # Batch inference
        inputs = [inputs] * args.num_prompts

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    print_outputs(outputs, "audio")

    image_inputs = get_multi_modal_input("image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-audios", type=int, help="The count of the audio", default=1)
    parser.add_argument("--num-prompts", type=int, help="The count of the audio", default=1)
    args = parser.parse_args()
    llm = LLM(model="Evo-LMM/kino-7b-qwen2_5_caps_conv", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    main(args)