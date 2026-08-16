# Memory Optimization Experiment Results

## Results

| Method | Peak Memory | Memory vs BF16 | Throughput | Speed vs BF16 | Trainable Params |
|---|---:|---:|---:|---:|---:|
| FP32 | 8.531 GB | +55.7% | 19,640 tok/s | 0.40x | 124.4M |
| **BF16 Baseline** | 5.478 GB | Baseline | 48,940 tok/s | 1.00x | 124.4M |
| Checkpoint | 3.496 GB | **-36.2%** | 41,562 tok/s | 0.85x | 124.4M |
| CPU Offloading | 5.078 GB | **-7.13%** | 14,427 tok/s | 2.25x | 124.4M |\
| Grad Accum | **2.158 GB** | **-60.6%** | 39,381 tok/s | 0.80x | 124.4M |
| 8-bit Adam | 5.236 GB | -4.4% | 37,995 tok/s | 0.78x | 124.4M |
| **LoRA** | 4.706 GB | -14.1% | **58,208 tok/s** | **1.19x** | 0.295M |
| LoRA + Checkpoint | 3.028 GB | **-44.7%** | 45,869 tok/s | 0.94x | 0.295M |
## Key Findings

### FP32 vs BF16
- Memory: **8.53 GB → 5.48 GB (-35.8%)**
- Throughput: **19.6K → 48.9K tok/s (2.49x)**
- BF16 provides a large improvement in both memory usage and training speed.

### Activation Checkpointing
- Memory: **5.48 GB → 3.50 GB (-36.2%)**
- Throughput: **48.9K → 41.6K tok/s (-15.1%)**
- Saves activation memory by recomputing activations during backward.
- Clear **memory vs. compute trade-off**.

### Gradient Accumulation
- Memory: **5.48 GB → 2.16 GB (-60.6%)**
- Throughput: **48.9K → 39.4K tok/s (-19.5%)**
- Lowest memory usage in this experiment.
- The memory reduction mainly comes from using a **smaller micro-batch** while maintaining the effective batch size through accumulation.

### 8-bit Adam
- Memory: **5.48 GB → 5.24 GB (-4.4%)**
- Throughput: **48.9K → 38.0K tok/s (-22.4%)**
- Limited benefit for this small 124M-parameter model.
- Expected to become more useful for larger models where optimizer states consume more memory.

### LoRA
- Memory: **5.48 GB → 4.71 GB (-14.1%)**
- Throughput: **48.9K → 58.2K tok/s (+19%)**
- Only **~0.24% of parameters are trainable**.
- Best throughput in this experiment.

### LoRA + Checkpoint
- Memory: **5.48 GB → 3.03 GB (-44.7%)**
- Throughput: **48.9K → 45.9K tok/s (-6.3%)**
- Good balance between memory reduction and training speed.

## Summary

- **Fastest:** LoRA (`58.2K tok/s`)
- **Lowest Memory:** Gradient Accumulation (`2.16 GB`)
- **Best Full Fine-tuning Trade-off:** Activation Checkpointing
- **Best LoRA Memory/Speed Trade-off:** LoRA + Checkpoint
- **Limited benefit on this small model:** 8-bit Adam

### Activation Checkpointing Result

> Activation checkpointing reduced peak GPU memory from **5.48 GB to 3.50 GB (-36%)**, while decreasing throughput from **48.9K to 41.6K tokens/sec (-15%)**.

This demonstrates the expected trade-off: **substantial memory savings at the cost of additional recomputation during backward propagation**.
