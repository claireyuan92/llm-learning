# Lesson 2: Embeddings + Positional Encoding (treasure: 向量)
# ---------------------------------------------------------------------------
# Sinusoidal positional encoding from scratch with torch:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
# Then we show that dot products of encodings reflect relative distance --
# nearby positions are similar, far ones are not. (Attention itself is
# permutation-invariant, so this is where "order" comes from.)
# Run: python lesson_02_embedding_posenc.py   (no args, CPU only)

import torch


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * -(torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def main():
    torch.manual_seed(0)
    d_model, seq = 32, 8
    pe = sinusoidal_pe(seq, d_model)

    print("PE slice, positions 0-3, dims 0-7:")
    print(pe[:4, :8].round(decimals=3))

    # similarity of position 0 with every other position
    sims = (pe[0:1] @ pe.T).squeeze(0)  # cosine-ish (rows ~ unit norm)
    print("\ndot(PE(0), PE(k)) for k = 0..7:")
    print(" ".join(f"{v:7.3f}" for v in sims.tolist()))

    # nearby positions stay similar, distant ones drift apart
    assert sims[1] > sims[6], "nearby positions should be more similar"
    print("\nPE(0) is most similar to itself, stays close to neighbors, "
          "drifts from far positions -- relative distance is encoded in the vectors.")
    print("Add this to token embeddings and the model can tell "
          "'dog chases cat' from 'cat chases dog'.")


if __name__ == "__main__":
    main()
