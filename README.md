# LLM Sampling Library

## Overview

The LLM Sampling Library is a Python package designed to facilitate text generation using various sampling methods with large language models (LLMs). This library provides a simple command-line interface (CLI) for users to generate text based on input prompts, utilizing models from the Hugging Face Transformers library.

## Features

- **Multiple Sampling Methods**: Implements various sampling techniques including:
  - Unconstrained sampling
  - Top-k sampling
  - Top-p (nucleus) sampling 
  - [Min-p sampling](https://arxiv.org/abs/2407.01082)
  - [Typical sampling](https://arxiv.org/abs/2202.00666)
  - [Epsilon sampling](https://arxiv.org/abs/2210.15191)
  - [Eta sampling](https://arxiv.org/abs/2210.15191)
  - Beam search
  - [Chain-of-Thought (CoT) decoding](https://arxiv.org/abs/2402.10200)
  - Constrained JSON decoding
  - [Speculative sampling](https://arxiv.org/abs/2302.01318)
  - [Medusa decoding](https://arxiv.org/abs/2401.10774)

- **Chat Template Support**: Optionally apply chat templates for Instruct models
- **Memory Efficient**: Uses KV-cache for better memory usage

In all implementations, temperature scaling is applied to the logits before any sampling methods, following the [GPT-2 implementation](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/sample.py#L61C5-L72C14) and [Hugging Face's implementation](https://github.com/huggingface/transformers/blob/acc394c4f5e1283c19783581790b3dc3105a3697/src/transformers/generation/utils.py#L825C9-L847C23).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python generate.py --model <model_name> --prompt "<input_prompt>" --apply-chat-template --temperature <temperature> --method <sampling_method> --max_new_tokens <max_new_tokens> --hf-token <hugging_face_token> --dtype <data_type>
```

### Parameters

- `--model`: The path or name of the Hugging Face model to use
- `--prompt`: The input sequence for the model
- `--prompt_file`: Alternative to --prompt, load prompt from a file
- `--temperature`: Sampling temperature (default: 1.0)
- `--method`: Sampling method to use (see list below)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 500)
- `--hf-token`: Your Hugging Face token for model access
- `--dtype`: Data type for model (bfloat16, float16, float32)
- `--seed`: Random seed for reproducibility

Method-specific parameters:
- `--top_k`: K value for top-k sampling
- `--top_p`: P value for nucleus sampling
- `--min_p`: Threshold for min-p sampling
- `--epsilon`: Epsilon value for epsilon/eta sampling
- `--beam_width`: Beam width for beam search
- `--typical_p_mass`: Mass parameter for typical sampling
- `--json_schema`: Schema file path for constrained JSON sampling
- `--draft-model`: Path to draft model for speculative sampling
- `--medusa-model-heads`: Path to Medusa model heads
- `--lookahead`: Lookahead parameter for speculative sampling

## Examples

Top-k sampling:
```bash
python generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Tell me a story" --method top_k --top_k 50 --temperature 0.7
```

Nucleus sampling:
```bash
python generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Write a poem" --method top_p --top_p 0.9 --temperature 0.8
```

Speculative sampling:
```bash
python generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Explain quantum physics" --method speculative --draft-model meta-llama/Llama-3.2-1B-Instruct --lookahead 4
```

Medusa decoding:
```bash
python generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Write code for merge sort" --method medusa --medusa-model-heads <path_to_heads>
```

### Acknowledgements

- [Medusa](https://github.com/FasterDecoding/Medusa) for the Medusa decoding implementation
- [vLLM](https://github.com/vllm-project/vllm) used as the reference for the Medusa heads architecture
- [Jsonformer](https://github.com/1rgs/jsonformer) for the constrained JSON decoding implementation
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for correctness evaluation
