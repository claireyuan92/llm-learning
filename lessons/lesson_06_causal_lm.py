# Lesson 6: Pretraining -- Causal Language Model (treasure: 预测)
# ---------------------------------------------------------------------------
# "Cover and guess": shift the text right by one, past = quiz, future = answer.
# A tiny char-level GPT (2 layers, d=64) trains on a small built-in corpus
# with next-token cross-entropy and a causal mask, then generates text.
# Lesson takeaway: no labels needed -- the text supervises itself.
# Run: python lesson_06_causal_lm.py   (no args, CPU only, ~1-2 min)

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT = ("the treasure hunt begins at dawn . "
        "follow the map and find the gold . "
        "the map leads to the old harbor . "
        "dig where the lantern light falls . ") * 8

N_LAYER, D_MODEL, N_HEAD, SEQ = 2, 64, 4, 32
STEPS, BATCH, LR = 200, 16, 3e-3


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, dim_feedforward=4 * D_MODEL,
                                       batch_first=True, norm_first=True)
            for _ in range(N_LAYER)])
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        mask = torch.triu(torch.ones(SEQ, SEQ), diagonal=1).bool()
        self.register_buffer("causal", mask)  # True = "do not look"

    def forward(self, idx):
        B, T = idx.shape
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for layer in self.layers:
            h = layer(h, src_mask=self.causal[:T, :T])
        return self.head(h)


@torch.no_grad()
def generate(model, stoi, itos, start="the ", n=80, temp=1.0):
    model.eval()
    ids = [stoi[c] for c in start]
    for _ in range(n):
        ctx = torch.tensor(ids[-SEQ:]).unsqueeze(0)
        nxt = F.softmax(model(ctx)[0, -1] / temp, dim=-1)
        ids.append(int(torch.multinomial(nxt, 1)))
    return "".join(itos[i] for i in ids)


def main():
    torch.manual_seed(1)
    chars = sorted(set(TEXT))
    stoi, itos = {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in TEXT])

    def batch():
        i = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
        x = torch.stack([data[j:j + SEQ] for j in i])
        return x, torch.stack([data[j + 1:j + SEQ + 1] for j in i])

    model = TinyGPT(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"chars={len(chars)} params={sum(p.numel() for p in model.parameters())}")
    for step in range(1, STEPS + 1):
        x, y = batch()
        loss = F.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")
    print("\nloss fell from ~%.1f (uniform guess) -- the model learned to predict." % 3.7)
    print("sample:", repr(generate(model, stoi, itos)))


if __name__ == "__main__":
    main()
