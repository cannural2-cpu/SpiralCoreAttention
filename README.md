# SpiralCoreAttention

## Experimental context selection for long-sequence LLM fine-tuning

SpiralCoreAttention is an experimental training-time context-selection project for open-weight language-model fine-tuning.

Before the training forward/backward pass, the prototype selects token blocks from a longer sequence. The goal is to reduce training-step computation and peak GPU memory while measuring the held-out full-context quality trade-off against a full-context baseline.

This is research software, not a production training system.

---

## What problem does this address?

Long-sequence LLM fine-tuning can be expensive because every training step processes the full input sequence.

This can increase:

- GPU memory use
- Training-step time
- Fine-tuning cost
- Limits on usable sequence length
- Experiment turnaround time

SpiralCoreAttention evaluates a specific question:

> Can a selected subset of training context reduce compute and memory cost while keeping the held-out full-context quality trade-off within an acceptable range?

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

It does not modify the base model’s attention kernel, preserve every unselected token at lower intensity, or replace the underlying language model.

---

## Main internal result: 7B QLoRA fine-tuning at 8K sequence length

The primary current result is a controlled internal fine-tuning experiment:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning method: 4-bit NF4 QLoRA
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Sequence length: 8,192 tokens
- Training steps: 48
- Context-selection configuration: `keep_ratio=0.60`, `min_keep=512`
- Dataset: local text corpus with held-out full-context evaluation
- Repeatability check: three seeds (`17`, `29`, `43`)

For each run, baseline and Spiral used the same base model, data split, optimizer, batch size, sequence length, and GPU environment.

| Seed | Training-step speedup | Peak VRAM saved | Held-out full-context loss difference |
|---|---:|---:|---:|
| 17 | 1.674× | 6.036 GB | -0.001137 |
| 29 | 1.675× | 6.036 GB | -0.004034 |
| 43 | 1.675× | 6.036 GB | -0.000861 |

### Summary across three internal runs

- Mean training-step speedup: approximately **1.675×**
- Equivalent training-step time reduction: approximately **40.3%**
- Peak VRAM reduction: **6.036 GB** in all three runs
- Mean held-out full-context loss difference: approximately **-0.0020**

Interpretation:

- Spiral completed training steps substantially faster in this specific internal setup.
- Peak GPU memory was consistently lower.
- Held-out full-context loss did not worsen across the three internal runs.
- These results are encouraging but do not establish universal quality retention.

---

## Earlier internal signals

### 7B QLoRA at 4K sequence length

On the same model family with 4K sequences and 64 training steps:

- Mean training-step speedup: approximately **1.608×**
- Peak VRAM reduction: **3.34 GB**
- Mean held-out full-context loss difference: approximately **+0.0123**
- Three internal seeds evaluated

### 1.5B QLoRA at 2K sequence length

An earlier internal pilot using `Qwen/Qwen2.5-1.5B-Instruct` on RTX 5090:

- Mean training-step speedup: **1.56×**
- Peak VRAM reduction: **1.224 GB**
- Held-out full-context loss difference: **+0.008957**

The 7B / 8K result is the primary current benchmark.

---

## What is validated today?

- Single-GPU QLoRA fine-tuning
- Qwen2.5 1.5B and 7B model configurations
- Training-time token-block context selection
- 2K, 4K, and 8K sequence-length experiments
- Controlled baseline-versus-Spiral comparisons
- Mean, p50, and p95 training-step timing
- Peak-VRAM reporting
- Held-out full-context loss reporting
- Three-seed repeatability checks on 7B experiments
- Local text-corpus evaluation

---

## What is not validated yet?

- Customer-specific datasets and task-quality metrics
- Longer-duration training convergence
- 14B, 32B, or 70B fine-tuning
- Multi-GPU or distributed training
- Full-model pretraining
- Production training infrastructure integration
- Guaranteed quality retention
- Guaranteed GPU-cost savings

Any customer decision should be based on a controlled validation using the target model, dataset, sequence length, training duration, hardware, and task-quality criteria.

---

## Reproducing the primary 7B / 8K pilot

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the baseline-versus-Spiral comparison:

```bash
python3 training/hf_lora_baseline_vs_spiral.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data test_data/realtext_50k.txt \
  --steps 48 \
  --batch-size 1 \
  --seq-len 8192 \
  --keep-ratio 0.60 \
  --min-keep 512 \
  --seed 17 \
  --output outputs/hf_qlora_7b_8192_seed17.json
```

Repeat with:

```bash
--seed 29
```

and:

```bash
--seed 43
```

The benchmark reports:

- Mean, p50, and p95 training-step timing
- Peak VRAM
- Retained-token ratio
- Held-out full-context evaluation loss
- Baseline-versus-Spiral comparison

Hardware, CUDA version, model version, dataset, sequence length, training duration, seed, quantisation settings, and selection configuration can materially affect results.

---

## No-cost controlled validation

SpiralCoreAttention is looking for a small number of technical design partners running long-sequence open-weight model fine-tuning.

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

If the validation shows value, a paid integration or broader evaluation can be discussed afterwards.

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

All benchmark figures in this repository are internal results from specific model, GPU, dataset, and configuration choices. Results may vary materially across models, sequence lengths, training methods, datasets, seeds, GPUs, and distributed-training setups.

Nothing in this repository should be interpreted as a guaranteed performance improvement, a guarantee of quality retention, or a claim about full-model pretraining or 70B-scale training.
