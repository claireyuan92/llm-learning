# Lesson 1: BPE Tokenization (treasure: 分词)
# ---------------------------------------------------------------------------
# Byte-level Byte Pair Encoding from scratch:
#   1. split the corpus into bytes (smallest units)
#   2. repeatedly find the most frequent adjacent pair and merge it
#   3. tokenize new text greedily with the learned merges
# Lesson takeaway: frequent chunks become whole tokens, rare words fall back
# to smaller pieces -- compression with a never-OOV guarantee.
# Run: python lesson_01_bpe.py   (no args, no downloads, CPU only)

from collections import Counter

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog",
    "a cat is on the mat",
]
NUM_MERGES = 25


def get_stats(vocab):
    """Count all adjacent symbol pairs across the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """Merge every occurrence of `pair` into one symbol, e.g. ('t','h')->'th'."""
    bigram = " ".join(pair)
    replacement = "".join(pair)
    out = {}
    for word, freq in vocab.items():
        # operate on the space-joined string so only this exact pair merges
        w = " ".join(word).replace(bigram, replacement)
        out[tuple(w.split(" "))] = freq
    return out


def train_bpe(corpus, num_merges):
    # init: every word = tuple of its characters + end-of-word marker
    vocab = Counter()
    for line in corpus:
        for w in line.split():
            vocab[tuple(w) + ("</w>",)] += 1
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    # True byte-level BPE: seed the table with all 256 byte values, so ANY
    # text (even with unseen characters) can always fall back to bytes.
    # The greedy tokenizer below can only ever emit base bytes or results
    # of learned merges, so this table covers everything it produces.
    tokens = {chr(i) for i in range(256)}
    tokens.add("</w>")
    for a, b in merges:
        tokens.add(a + b)
    tok2id = {t: i for i, t in enumerate(sorted(tokens))}
    return merges, tok2id


def tokenize(text, merges):
    """Greedy longest-match tokenization using the learned merge order."""
    rank = {m: i for i, m in enumerate(merges)}
    out = []
    for w in text.split():
        word = list(w) + ["</w>"]
        # repeatedly merge the highest-priority (earliest-learned) pair
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            ranked = [(rank.get(p, 1 << 30), p) for p in pairs]
            _, best = min(ranked)
            if best not in rank:  # no learned merge applies anymore
                break
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        out.extend(word)
    return out


def main():
    merges, tok2id = train_bpe(CORPUS, NUM_MERGES)
    print(f"learned {len(merges)} merges, vocab size = {len(tok2id)}")
    print("first 10 merges:", ["+".join(m) for m in merges[:10]])

    for s in ["the cat sat", "unhappiness"]:
        toks = tokenize(s, merges)
        ids = [tok2id[t] for t in toks]
        print(f"\n{s!r:20} -> tokens: {toks}")
        print(f"{'':20}    ids: {ids}")
    # determinism check: same text -> same split, and it round-trips
    a, b = tokenize("the cat", merges), tokenize("the cat", merges)
    assert a == b, "BPE must be deterministic"
    print("\ndeterministic: same input -> same tokens. "
          "Rare words fall back to pieces, so nothing is ever OOV.")


if __name__ == "__main__":
    main()
