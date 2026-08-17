import gc
import time
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL = "gpt2"

SEQ_LEN = 512

# baseline effective batch = 8
EFFECTIVE_BATCH = 8

WARMUP = 3
STEPS = 10

DEVICE = "cuda"


CONFIGS = [
    {
        "name": "FP32",
        "dtype": torch.float32,
        "batch": 8,
        "grad_accum": 1,
    },
    {
        "name": "BF16 Baseline",
        "dtype": torch.bfloat16,
        "batch": 8,
        "grad_accum": 1,
    },
    {
        "name": "Checkpoint",
        "dtype": torch.bfloat16,
        "batch": 8,
        "grad_accum": 1,
        "checkpoint": True,
    },
    {
        "name": "Grad Accum",
        "dtype": torch.bfloat16,
        "batch": 2,
        "grad_accum": 4,
    },
    {
        "name": "8-bit Adam",
        "dtype": torch.bfloat16,
        "batch": 8,
        "grad_accum": 1,
        "adam8bit": True,
    },
    {
        "name": "LoRA",
        "dtype": torch.bfloat16,
        "batch": 8,
        "grad_accum": 1,
        "lora": True,
    },
    {
        "name": "LoRA+Checkpoint",
        "dtype": torch.bfloat16,
        "batch": 8,
        "grad_accum": 1,
        "lora": True,
        "checkpoint": True,
    },
]


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def create_model(cfg):

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=cfg["dtype"],
    ).to(DEVICE)

    model.config.use_cache = False

    if cfg.get("lora", False):

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(
            model,
            lora_config,
        )

    if cfg.get("checkpoint", False):
        model.gradient_checkpointing_enable()

        # LoRA + gradient checkpointing에서 필요할 수 있음
        if cfg.get("lora", False):
            model.enable_input_require_grads()

    model.train()

    return model


def create_optimizer(model, cfg):

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if cfg.get("adam8bit", False):

        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=1e-4,
        )

    else:

        optimizer = torch.optim.AdamW(
            params,
            lr=1e-4,
        )

    return optimizer


def train_one_effective_step(
    model,
    optimizer,
    batch_size,
    grad_accum,
):

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0

    for _ in range(grad_accum):

        x = torch.randint(
            0,
            model.config.vocab_size,
            (batch_size, SEQ_LEN),
            device=DEVICE,
        )

        out = model(
            input_ids=x,
            labels=x,
        )

        loss = out.loss / grad_accum

        loss.backward()

        total_loss += loss.item()

    optimizer.step()

    return total_loss


def benchmark(cfg):

    cleanup()

    print("\n" + "=" * 70)
    print(cfg["name"])
    print("=" * 70)

    model = create_model(cfg)

    optimizer = create_optimizer(
        model,
        cfg,
    )

    batch_size = cfg["batch"]
    grad_accum = cfg["grad_accum"]

    # --------------------
    # Warmup
    # --------------------

    for _ in range(WARMUP):

        train_one_effective_step(
            model,
            optimizer,
            batch_size,
            grad_accum,
        )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # --------------------
    # Benchmark
    # --------------------

    start = time.perf_counter()

    losses = []

    for _ in range(STEPS):

        loss = train_one_effective_step(
            model,
            optimizer,
            batch_size,
            grad_accum,
        )

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

    sec_per_step = elapsed / STEPS

    # 모든 config에서 effective batch 동일
    tokens = (
        EFFECTIVE_BATCH
        * SEQ_LEN
        * STEPS
    )

    tokens_per_sec = tokens / elapsed

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    result = {
        "name": cfg["name"],
        "memory": peak_allocated,
        "reserved": peak_reserved,
        "sec_step": sec_per_step,
        "tokens_sec": tokens_per_sec,
        "trainable": trainable,
        "total_params": total_params,
        "loss": sum(losses) / len(losses),
    }

    print(
        f"""
Peak allocated : {peak_allocated:.3f} GB
Peak reserved  : {peak_reserved:.3f} GB
sec/step       : {sec_per_step:.4f}
tokens/sec     : {tokens_per_sec:.1f}
trainable      : {trainable:,}
total params   : {total_params:,}
"""
    )

    del model
    del optimizer

    cleanup()

    return result


# ============================================================
# RUN
# ============================================================

results = []

for cfg in CONFIGS:

    try:

        r = benchmark(cfg)

        results.append(r)

    except Exception as e:

        print(
            f"\nFAILED: {cfg['name']}"
        )

        print(e)

        cleanup()


# ============================================================
# TABLE
# ============================================================

print("\n")
print("=" * 105)
print("FINAL RESULTS")
print("=" * 105)

print(
    f"{'Method':<22}"
    f"{'Memory GB':>12}"
    f"{'Reserved':>12}"
    f"{'sec/step':>12}"
    f"{'tok/sec':>14}"
    f"{'Trainable':>16}"
)

print("-" * 105)

for r in results:

    print(
        f"{r['name']:<22}"
        f"{r['memory']:>12.3f}"
        f"{r['reserved']:>12.3f}"
        f"{r['sec_step']:>12.4f}"
        f"{r['tokens_sec']:>14.1f}"
        f"{r['trainable']:>16,}"
    )


# ============================================================
# GRAPH
# ============================================================

names = [r["name"] for r in results]


# ---------------------------
# GPU memory
# ---------------------------

plt.figure(figsize=(11, 6))

values = [
    r["memory"]
    for r in results
]

plt.bar(names, values)

plt.ylabel("Peak GPU Memory (GB)")
plt.title("Training GPU Memory")

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    "memory_comparison.png",
    dpi=150,
)

plt.close()


# ---------------------------
# Speed
# ---------------------------

plt.figure(figsize=(11, 6))

values = [
    r["tokens_sec"]
    for r in results
]

plt.bar(names, values)

plt.ylabel("Tokens / sec")
plt.title("Training Throughput")

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    "throughput_comparison.png",
    dpi=150,
)

plt.close()


# ---------------------------
# Memory vs Speed
# ---------------------------

plt.figure(figsize=(9, 6))

for r in results:

    plt.scatter(
        r["memory"],
        r["tokens_sec"],
        s=100,
    )

    plt.annotate(
        r["name"],
        (
            r["memory"],
            r["tokens_sec"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.xlabel(
    "Peak GPU Memory (GB)"
)

plt.ylabel(
    "Tokens / sec"
)

plt.title(
    "Memory vs Training Throughput"
)

plt.tight_layout()

plt.savefig(
    "memory_vs_speed.png",
    dpi=150,
)

plt.close()


print("\nSaved:")
print("  memory_comparison.png")
print("  throughput_comparison.png")
print("  memory_vs_speed.png")
