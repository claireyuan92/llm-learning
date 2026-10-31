# Sampling Playground

Interactive playground for decoding-time sampling: temperature, top-k, top-p.

## Goal
Build intuition for how sampling hyperparameters shape token distributions.

## Milestones
- [x] Visualize how temperature sharpens / flattens the distribution
- [x] Compare top-k vs top-p truncation on a real logits vector
- [x] Add reproducible experiments for the Lesson 7 concepts

## Run

Requires Python 3 and NumPy:

```bash
python3 -m pip install numpy
python3 test_sampling.py
python3 __main__.py
python3 __main__.py --temperature 0.3 --samples 1000
python3 __main__.py --top-k 3 --samples 1000 --seed 42
python3 __main__.py --top-p 0.9 --samples 1000
```

The output shows the normalized sampling distribution and its entropy. With
`--samples`, it also runs a seeded categorical sampler so stochastic results
can be reproduced.

## Example temperature effects

```bash
python3 __main__.py --temperature 0.5
python3 __main__.py --temperature 1.0
python3 __main__.py --temperature 2.0
```
