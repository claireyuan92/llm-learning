# How to run everything

## One-time setup

1. Clone the repo:
   ```
   git clone https://github.com/claireyuan92/llm-learning.git
   cd llm-learning
   ```
2. Use Python 3.10+:
   ```
   python3 --version
   ```
3. Install the basics (CPU-only torch is fine for all lesson demos):
   ```
   pip install torch numpy
   ```
4. GPU note: lessons 1–9 and the sampling/KV-cache/quantization projects run on CPU.
   Triton, vLLM, and DDP/FSDP need a real NVIDIA GPU — use a cloud VM
   (e.g. Lambda Labs, RunPod) or a local NVIDIA card. The lesson-10 script
   detects CUDA and falls back to a CPU demo automatically.

## Lesson demos (`lessons/`)

Each script is standalone — just run it:

```
python lessons/lesson_01_bpe.py            # BPE tokenization from scratch
python lessons/lesson_02_embedding_posenc.py  # sinusoidal positional encoding
python lessons/lesson_03_attention.py      # scaled dot-product attention
python lessons/lesson_04_multihead.py      # multi-head attention
python lessons/lesson_05_transformer_block.py  # full transformer block
python lessons/lesson_06_causal_lm.py      # tiny char-level GPT training
python lessons/lesson_07_sampling.py       # temperature / top-k / top-p
python lessons/lesson_08_lora.py           # LoRA layer from scratch
python lessons/lesson_09_profiler.py       # attention timing vs seq length
python lessons/lesson_10_gpu_arch.py       # GPU memory-hierarchy demo
```

Each finishes in under ~2 minutes on CPU and prints what it demonstrates.

## Hands-on projects (`projects/`)

| Project | How to run | Needs GPU? | Status |
|---|---|---|---|
| 01-sampling-playground | code TBD | No | README + milestones |
| 02-kv-cache | code TBD (`pip install torch` only) | No | README + milestones |
| 03-triton-kernel | code TBD (`pip install triton`) | Yes, NVIDIA | README + milestones |
| 04-quantization | code TBD (`pip install torch`) | No for hand-rolled INT8 | README + milestones |
| 05-vllm-serving-benchmark | code TBD (`pip install vllm`) | Yes, NVIDIA | README + milestones |
| 06-ddp-fsdp | code TBD (`torchrun --nproc_per_node=2 ...`) | Yes, 2+ GPUs | README + milestones |

"Code TBD" means the project folder currently holds the goal and milestones;
starter code lands as the learning journey reaches that topic.
