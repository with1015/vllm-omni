# HyperCLOVAX-SEED-Omni-8B with vLLM-Omni

[HyperCLOVAX-SEED-Omni-8B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B)
is an omni-modal model by NAVER Cloud that supports:

| Input  | Output          |
|--------|-----------------|
| Text   | Text            |
| Audio  | Text + Audio    |
| Image  | Text            |
| Text   | Text + Image    |
| Audio  | Text + Audio + Image |

## Architecture

The model uses a 3-stage pipeline:

```
Stage 0 (Thinker) ──→ Stage 1 (Vision Decoder, diffusion)
         │
         └──────────→ Stage 2 (Audio Decoder, unit-BigVGAN)
```

- **Thinker**: Qwen2.5-VL vision encoder + Qwen2Audio encoder + HyperCLOVAX language model.
  Outputs text tokens and discrete audio/vision codes in the vocabulary.
- **Vision Decoder**: Diffusion-based image generation from 729 discrete TA-Tok codes.
- **Audio Decoder**: Unit-BigVGAN vocoder from CosyVoice2 FSQ discrete audio codes.

## Hardware Requirements

| Setup     | GPUs                                        |
|-----------|---------------------------------------------|
| Default   | 6 × GPU ≥24 GB (4 for thinker TP, 1+1 for decoders) |
| Minimal   | 3 × GPU ≥24 GB (1 for thinker, 1+1 for decoders) |

## Quick Start

### 1. Start the Server

```bash
# 6-GPU setup (production)
./run_server.sh --model naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B

# Custom GPU allocation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_server.sh
```

### 2. Run the Client Demo

```bash
# All modes: text-only, text-to-vision, speech-to-speech
python client_demo.py --base-url http://localhost:8000/v1

# Speech-to-Speech with your own audio file
python client_demo.py --mode s2s --audio-file /path/to/speech.wav

# Text-to-Vision
python client_demo.py --mode t2v --prompt "고양이 그림을 그려줘"
```

### 3. Use the OpenAI API Directly

**Speech-to-Speech:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
    "modalities": ["text", "audio"],
    "messages": [{
      "role": "user",
      "content": [
        {"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}},
        {"type": "text", "text": "이 오디오에 무슨 내용이 있나요?"}
      ]
    }]
  }'
```

**Text-to-Vision:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
    "modalities": ["text", "image"],
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "귀여운 강아지 한 마리가 공원에서 뛰노는 그림을 그려줘."}
      ]
    }]
  }'
```

## Stage Config

The default stage config is at
`vllm_omni/model_executor/stage_configs/hcx_omni.yaml`.

Key parameters:

| Stage | Type      | `model_arch` / `model_class_name`  | GPU   |
|-------|-----------|------------------------------------|-------|
| 0     | LLM       | `HCXVisionV2ForCausalLM`           | 0-3   |
| 1     | Diffusion | `HyperCLOVAXVisionPipeline`        | 4     |
| 2     | Diffusion | `HyperCLOVAXAudioPipeline`         | 5     |
