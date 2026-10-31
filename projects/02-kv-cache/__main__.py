import numpy as np

from kv_cache import DecoderConfig, ToyDecoder, benchmark_decode, kv_cache_bytes


def human_bytes(num_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} B"


def main() -> None:
    config = DecoderConfig(vocab_size=64, model_dim=32, num_heads=4)
    decoder = ToyDecoder(config)
    prompt = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    new_tokens = 12

    naive, cached = benchmark_decode(decoder, prompt, new_tokens, repeats=5)
    naive_tokens = decoder.decode_naive(prompt, new_tokens)
    cached_tokens = decoder.decode_cached(prompt, new_tokens)
    np.testing.assert_array_equal(naive_tokens, cached_tokens)

    bytes_needed = kv_cache_bytes(
        batch_size=1,
        sequence_length=prompt.shape[1] + new_tokens,
        num_layers=1,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
    )

    print("naive tokens:", naive_tokens.tolist())
    print("cached tokens:", cached_tokens.tolist())
    print(f"naive decode: {naive * 1000:.3f} ms")
    print(f"cached decode: {cached * 1000:.3f} ms")
    print(f"end-to-end speedup: {naive / cached:.2f}x")
    print(f"KV cache memory: {human_bytes(bytes_needed)}")


if __name__ == "__main__":
    main()
