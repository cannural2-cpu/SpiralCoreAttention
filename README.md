# SpiralCoreAttention

## Experimental context selection for long-sequence LLM fine-tuning

SpiralCoreAttention is an experimental training-time context-selection project for open-weight language-model fine-tuning.

Before the training forward/backward pass, the prototype selects token blocks from a longer sequence. The goal is to reduce training-step computation and peak GPU memory while measuring whether held-out full-context quality remains acceptable.

This is research software, not a production training system.

---

## What problem does this address?

Long-sequence LLM fine-tuning can be expensive because every training step processes the full input sequence.

This may increase:

- GPU memory use
- Training-step time
- Fine-tuning cost
- Limits on usable sequence length
- Experiment turnaround time

SpiralCoreAttention evaluates a simple question:

> Can a selected subset of training context reduce compute and memory cost while retaining acceptable held-out evaluation quality?

---

## How it works

```text
Long training sequence
        ↓
Token-block context selection
        ↓
Reduced selected sequence
        ↓
Training forward / backward pass
        ↓
Held-out full-context evaluation
```

The current prototype physically selects token blocks before the training pass.

It does not claim to modify the underlying model’s attention kernel, preserve every dropped token at lower intensity, or replace the base model.

---

## Current internal QLoRA validation

The main current result is an internal controlled fine-tuning experiment:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: 4-bit QLoRA
- GPU: NVIDIA GeForce RTX 5090
- Sequence length: 2,048 tokens
- Training steps: 64
- Dataset: local text corpus with held-out full-context evaluation
- Baseline and Spiral use the same base model, adapter setup, data split, optimizer, and GPU environment.

| Metric | Baseline | Spiral |
|---|---:|---:|
| Mean training-step speed | 1.00× | 1.56× |
| p50 training-step speed | 1.00× | 1.55× |
| Peak VRAM | Reference | 1.224 GB lower |
| Held-out full-context loss difference | — | +0.008957 |

Interpretation:

- Spiral completed mean training steps approximately **1.56× faster** in this internal configuration.
- Spiral used **1.224 GB less peak VRAM**.
- The held-out full-context loss difference was **+0.008957** in this run.

These are early internal measurements from one QLoRA configuration. They are not a production guarantee or a claim about larger models, multi-GPU training, customer workloads, or full-model pretraining.

---

## Additional controlled training signal

A smaller custom Transformer benchmark was also run at 4,096 sequence length on RTX 5090.

Across internal runs using a local text corpus:

- Approximately **2.09× mean training-step speedup**
- Approximately **2.01× p50 training-step speedup**
- **0.97 GB** lower peak VRAM
- Held-out full-context loss differences from approximately **+0.0048 to +0.0088**

This supports the research hypothesis, but it is not a substitute for validation on a target model and workload.

---

## What is validated today?

- Single-GPU fine-tuning on RTX 5090
- Qwen2.5-1.5B-Instruct 4-bit QLoRA integration
- Training-time token-block selection
- Baseline-versus-Spiral controlled comparison
- Training-step timing and peak-VRAM reporting
- Held-out full-context loss reporting
- Local text-corpus evaluation

---

## What is not validated yet?

- 7B, 14B, 32B, or 70B fine-tuning
- Multi-GPU or distributed training
- Full-model pretraining
- Customer-specific datasets
- Long-duration training convergence
- Production training infrastructure integration
- Guaranteed quality retention
- Guaranteed GPU-cost savings

Any customer decision should be based on a controlled validation using that customer’s target model, sequence lengths, dataset, quality criteria, and hardware.

---

## Reproducing the current QLoRA pilot

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the QLoRA comparison:

```bash
python3 training/hf_lora_baseline_vs_spiral.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data test_data/realtext_50k.txt \
  --steps 64 \
  --batch-size 1 \
  --seq-len 2048 \
  --keep-ratio 0.60 \
  --min-keep 256 \
  --output outputs/hf_qlora_64step.json
```

The script reports:

- Mean, p50, and p95 training-step timing
- Peak VRAM
- Retained-token ratio
- Held-out full-context evaluation loss
- Baseline-versus-Spiral comparison

Hardware, CUDA version, model version, dataset, training duration, seed, quantisation settings, and sequence length can materially affect results.

---

## No-cost controlled validation

SpiralCoreAttention is looking for a small number of technical design partners that run long-sequence open-weight model fine-tuning.

The initial validation is no-cost and fixed-scope.

Proposed scope:

- One approved fine-tuning workload
- Customer-owned or customer-approved environment
- Same model, data split, seed, optimizer, and GPU setup for baseline and Spiral
- Local or anonymised representative data
- Training-step latency measurement
- Peak-VRAM measurement
- Retained-token reporting
- Held-out full-context loss comparison
- No production access required
- No raw customer-data export required
- No obligation to continue if the result is not useful

If the controlled validation shows value, a paid integration or broader evaluation can be discussed afterwards.

---

## Website

https://spiralcoreattention.com

---

## Contact

Can Yılmaz Nural  
Founder, SpiralCoreAttention

Email: cannural.contact@gmail.com

---

## Disclaimer

SpiralCoreAttention is experimental research software.

All benchmark figures in this repository are internal results from specific model, hardware, dataset, and configuration choices. Results may vary materially across models, sequence lengths, training methods, datasets, seeds, GPUs, and distributed-training setups.

Nothing in this repository should be interpreted as a guaranteed performance improvement, a guarantee of quality retention, or a claim about full-model pretraining or 70B-scale training.
