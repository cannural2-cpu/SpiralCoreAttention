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

## Primary internal result: full context vs random blocks vs Spiral

The primary result is a controlled internal QLoRA comparison designed to separate generic shortening from value created by the selector.

### Setup

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning method: 4-bit NF4 QLoRA
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Sequence length: 8,192 tokens
- Training steps: 48
- Reduced-context configuration: `keep_ratio=0.60`, `min_keep=512`, `block_size=64`
- Dataset: local text corpus with held-out full-context evaluation
- Seeds: `17`, `29`, `43`

For every seed, the following conditions used the same model, initial adapter state, data split, optimizer, batch size, sequence length, and GPU environment:

1. Full context: 100%
2. Matched random block selection: approximately 60%
3. Spiral-selected block selection: approximately 60%

End-to-end step timing includes selection work.

### Mean results across three seeds

| Condition | Held-out full-context loss | Mean step time | Peak VRAM |
|---|---:|---:|---:|
| Full context 100% | 0.443981 | 2687.78 ms | 25.998 GB |
| Random blocks ~60% | 0.445061 | 1613.56 ms | 19.956 GB |
| Spiral-selected ~60% | 0.444474 | 1611.53 ms | 19.958 GB |

### Derived comparisons

| Comparison | Result |
|---|---:|
| Spiral speedup versus full context | 1.668× |
| Spiral peak VRAM reduction versus full context | 6.040 GB |
| Random loss difference versus full context | +0.001080 |
| Spiral loss difference versus full context | +0.000493 |
| Spiral loss difference versus random blocks | -0.000587 |

Lower loss is better.

### Interpretation

The controlled result supports three conclusions:

- Reducing the processed context to approximately 60% produced a substantial efficiency gain in this local 8K QLoRA setup.
- Generic shortening explains most of that efficiency gain: random blocks and Spiral-selected blocks had very similar step time and memory use.
- Spiral-selected blocks had slightly lower held-out full-context loss than matched random blocks in all three seeds.

The quality difference between Spiral and random selection is small. It is therefore an encouraging repeatable signal, not proof that the selector is generally superior across datasets or workloads.

---

## Checkpointed full-context evaluation

Full-context held-out loss was measured at steps 0, 12, 24, and 48.

| Step | Full context | Random ~60% | Spiral ~60% |
|---|---:|---:|---:|
| 0 | 0.528952 | 0.528952 | 0.528952 |
| 12, mean across seeds | 0.472666 | 0.465087 | 0.464725 |
| 24, mean across seeds | 0.447958 | 0.448567 | 0.448271 |
| 48, mean across seeds | 0.443981 | 0.445061 | 0.444474 |

These short-run curves show that all three conditions learned under the same evaluation protocol. They do not establish longer-duration convergence behavior.

---

## Experimental contract

The current controlled benchmark makes the following implementation choices explicit:

- Full, random, and Spiral conditions begin from the same LoRA adapter state.
- Selection/scoring time is included in end-to-end step time.
- Selected blocks use compacted position IDs after selection.
- The first loss-bearing target after a non-adjacent block join is masked by default.
- Every condition is evaluated on the original, unmodified held-out full-context sequences.
- Random selection preserves the same block granularity and approximate retained-token budget as Spiral.

These details matter because physically joining non-adjacent blocks changes sequence topology.

---

## Current follow-up work

The next evaluation additions are:

- A controlled cross-block dependency coverage probe, asking whether both distant evidence blocks needed for a target are retained.
- Public deterministic evaluation inputs for protocol review.
- Task-level quality evaluation beyond aggregate language-model loss.
- Longer training runs and additional datasets.
- Additional model sizes and distributed-training configurations.

The cross-block probe is a mechanism test, not a substitute for realistic downstream task evaluation.

---

## Earlier internal signals

Earlier internal experiments showed similar efficiency signals, but they are preliminary and should not be numerically combined with the primary controlled three-condition result above.

### 7B QLoRA at 4K sequence length

- Mean training-step speedup: approximately **1.608×**
- Peak VRAM reduction: **3.34 GB**
- Three internal seeds evaluated

### 1.5B QLoRA at 2K sequence length

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: RTX 5090
- Mean training-step speedup: **1.56×**
- Peak VRAM reduction: **1.224 GB**

---

## What is validated today?

- Single-GPU QLoRA fine-tuning
- Qwen2.5 1.5B and 7B model configurations
- Training-time token-block context selection
- 2K, 4K, and 8K sequence-length experiments
- Full-context versus random-block versus Spiral-selected controlled comparison
- Three-seed repeatability check at 7B / 8K
- End-to-end training-step timing including selection work
- Peak-VRAM reporting
- Held-out full-context loss reporting
- Checkpointed evaluation at steps 0, 12, 24, and 48
- Local text-corpus evaluation

---

## What is not validated yet?

- Public reproduction of the exact internal corpus result
- Customer-specific datasets and task-quality metrics
- Longer-duration training convergence
- Downstream task accuracy
- 14B, 32B, or 70B fine-tuning
- Multi-GPU or distributed training
- Full-model pretraining
- Production training infrastructure integration
- Guaranteed quality retention
- Guaranteed GPU-cost savings

Any customer decision should be based on a controlled validation using the target model, dataset, sequence length, training duration, hardware, and task-quality criteria.

---

## Running the controlled benchmark

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The benchmark accepts a local plain-text corpus. The exact internal corpus used for the reported numbers is not currently published, so external users should treat their run as a protocol reproduction, not an exact numerical reproduction.

```bash
python3 training/hf_lora_baseline_vs_spiral.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data /path/to/local_corpus.txt \
  --steps 48 \
  --batch-size 1 \
  --seq-len 8192 \
  --keep-ratio 0.60 \
  --min-keep 512 \
  --block-size 64 \
  --checkpoint-steps 0,12,24,48 \
  --seed 17 \
  --output outputs/controlled_7b_8k_seed17.json
```

Repeat with seeds `29` and `43`.

The report includes:

- Full-context, random-block, and Spiral-selected conditions
- Checkpointed full-context evaluation loss
- End-to-end step time
- Selection/scoring time
- Peak VRAM
- Retained-token ratio
- Position-ID policy
- Block-join loss policy

Hardware, CUDA version, model version, dataset, sequence length, training duration, seed, quantisation settings, and selection configuration can materially affect results.

---

## No-cost controlled validation

SpiralCoreAttention is looking for a small number of technical design partners running long-sequence open-weight model fine-tuning.

The initial validation is no-cost and fixed-scope.

Proposed scope:

- One approved fine-tuning workload
- Customer-owned or customer-approved environment
- Same mod
