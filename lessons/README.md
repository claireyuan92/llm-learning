# LLM Learning Journey — Lesson Code Index
# 10 self-contained runnable demos, one per completed lesson (1–10).

| Lesson | Script | Treasure | What it demonstrates |
|---|---|---|---|
| 1 | `lesson_01_bpe.py` | 分词 | Byte-level BPE trained from scratch; tokenize + ids |
| 2 | `lesson_02_embedding_posenc.py` | 向量 | Sinusoidal positional encoding; dot products reflect distance |
| 3 | `lesson_03_attention.py` | 注意力 | Scaled dot-product attention, verified vs torch; causal mask |
| 4 | `lesson_04_multihead.py` | 多头 | MHA from scratch, verified vs `nn.MultiheadAttention`; per-head weights |
| 5 | `lesson_05_transformer_block.py` | 残差 | Full block (MHA+MLP+residual+LayerNorm); shape test; gradient flows via residual |
| 6 | `lesson_06_causal_lm.py` | 预测 | Tiny char-level GPT, next-token pretraining on built-in corpus; loss ↓ + sample |
| 7 | `lesson_07_sampling.py` | 随机 | temperature / top-k / top-p on fixed logits; sampled tokens |
| 8 | `lesson_08_lora.py` | 微调 | LoRALinear from scratch; param counts; one step updates only A/B |
| 9 | `lesson_09_profiler.py` | 瓶颈 | Attention timing vs seq len; quadratic growth table |
| 10 | `lesson_10_gpu_arch.py` | 并行 | Coalesced vs strided memory bandwidth (CUDA) or CPU cache-locality demo |

## How to run

```bash
python lesson_01_bpe.py
# ... or any of the ten; each takes no arguments
```

- Dependencies: `torch` + `numpy` only (`pip install torch numpy`).
- No downloads, no data files, no GPU required — everything runs on CPU.
- Each script finishes in well under 2 minutes on CPU.
- CUDA code paths (lesson 10) are guarded by `torch.cuda.is_available()`.
