# DDP vs FSDP

Distributed training: from data-parallel basics to sharded training.

## Goal
Understand how models bigger than one GPU get trained: gradient sync (DDP)
vs parameter sharding (FSDP).

## Milestones
- [ ] DDP training on 2+ GPUs; measure scaling efficiency
- [ ] Same workload with FSDP; compare peak memory
- [ ] Profile the all-reduce / all-gather communication overhead
