import gc
import time
import torch

from transformers import AutoModelForCausalLM
from accelerate import cpu_offload


MODEL = "gpt2"

BATCH_SIZE = 8
SEQ_LEN = 512

WARMUP_STEPS = 3
MEASURE_STEPS = 10

DEVICE = "cuda"


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def make_batch(model):
    x = torch.randint(
        0,
        model.config.vocab_size,
        (BATCH_SIZE, SEQ_LEN),
        device=DEVICE,
    )
    return x


def train_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)

    x = make_batch(model)

    out = model(
        input_ids=x,
        labels=x,
    )

    loss = out.loss
    loss.backward()
    optimizer.step()

    return loss.item()


def run_baseline():
    cleanup()

    print("\n" + "=" * 70)
    print("BF16 BASELINE")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
    ).to(DEVICE)

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # warmup
    for _ in range(WARMUP_STEPS):
        train_step(model, optimizer)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    losses = []

    for _ in range(MEASURE_STEPS):
        loss = train_step(model, optimizer)
        losses.append(loss)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    sec_per_step = elapsed / MEASURE_STEPS

    tokens_per_sec = (
        BATCH_SIZE
        * SEQ_LEN
        * MEASURE_STEPS
        / elapsed
    )

    result = {
        "name": "BF16 Baseline",
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "sec_per_step": sec_per_step,
        "tokens_per_sec": tokens_per_sec,
        "loss": sum(losses) / len(losses),
    }

    del optimizer
    del model

    cleanup()

    return result


def run_cpu_offload():
    cleanup()

    print("\n" + "=" * 70)
    print("CPU OFFLOAD")
    print("=" * 70)

    # 처음에는 CPU에 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False
    model.train()

    # 필요한 module만 forward 시 GPU로 이동
    cpu_offload(
        model,
        execution_device=torch.device(DEVICE),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # warmup
    for _ in range(WARMUP_STEPS):
        train_step(model, optimizer)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    losses = []

    for _ in range(MEASURE_STEPS):
        loss = train_step(model, optimizer)
        losses.append(loss)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    sec_per_step = elapsed / MEASURE_STEPS

    tokens_per_sec = (
        BATCH_SIZE
        * SEQ_LEN
        * MEASURE_STEPS
        / elapsed
    )

    result = {
        "name": "CPU Offload",
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "sec_per_step": sec_per_step,
        "tokens_per_sec": tokens_per_sec,
        "loss": sum(losses) / len(losses),
    }

    del optimizer
    del model

    cleanup()

    return result


def print_result(results):
    print("\n" + "=" * 90)
    print("RESULT")
    print("=" * 90)

    print(
        f"{'Method':<20}"
        f"{'Peak Alloc GB':>15}"
        f"{'Peak Reserved':>15}"
        f"{'sec/step':>15}"
        f"{'tokens/sec':>15}"
    )

    print("-" * 90)

    for r in results:
        print(
            f"{r['name']:<20}"
            f"{r['peak_allocated']:>15.3f}"
            f"{r['peak_reserved']:>15.3f}"
            f"{r['sec_per_step']:>15.4f}"
            f"{r['tokens_per_sec']:>15.1f}"
        )

    baseline = results[0]
    offload = results[1]

    memory_saved = (
        1
        - offload["peak_allocated"]
        / baseline["peak_allocated"]
    ) * 100

    slowdown = (
        offload["sec_per_step"]
        / baseline["sec_per_step"]
        - 1
    ) * 100

    print("\nComparison")
    print("-" * 40)
    print(f"GPU memory saved : {memory_saved:.2f}%")
    print(f"Slowdown         : {slowdown:.2f}%")


if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    baseline = run_baseline()
    offload = run_cpu_offload()

    print_result([
        baseline,
        offload,
    ])
