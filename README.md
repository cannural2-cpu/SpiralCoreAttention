# SpiralCoreAttention

## Experimental context selection for long-sequence LLM fine-tuning

SpiralCoreAttention is research software for **training-time context selection** in open-weight LLM fine-tuning. Before a training forward/backward pass, it selects a subset of candidate context paragraphs or token blocks. The aim is to reduce step time and peak GPU memory while measuring whether held-out, full-context task quality is preserved.

This is not a production training system. It does not modify a model's attention kernel, and discarded context is not retained at lower intensity: the prototype physically selects context before the training pass.

## The question

> At a fixed reduced-context budget, can task-aware selection preserve evidence needed for long-context fine-tuning better than random shortening, while reducing compute and memory?

A speedup alone is not enough. A shorter context may appear safe simply because a dataset is redundant, so every meaningful selection policy needs a matched random control.

## Primary internal evidence: MuSiQue 2-hop QLoRA

The primary result is a controlled three-seed fine-tuning pilot on the dependency-rich [MuSiQue](https://github.com/StonyBrookNLP/musique) 2-hop QA task. Every condition is evaluated on the same **full-context** held-out QA prompts; only its training-time paragraph-selection policy changes.

### Setup

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning: 4-bit NF4 QLoRA
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Training: 96 MuSiQue 2-hop cases, 192 optimizer steps, batch size 1
- Held-out evaluation: 80 MuSiQue 2-hop cases with full context
- Context budget: 60% of candidate paragraphs; paragraphs capped at 128 tokens
- Seeds: 17, 29, and 43

All conditions used the same base model, initial adapter weights, optimizer configuration, data order, batch size, sequence budget, and reset dropout RNG stream.

| Training policy | Held-out full-context F1 | Exact match | Mean step time | Peak VRAM | Train cases retaining all gold support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full context | 0.4119 | 0.2875 | 684 ms | 13.214 GB | 100.0% |
| Matched random 60% | 0.3421 | 0.2125 | 406 ms | 11.130 GB | 37.5% |
| Cheap gate 60% | 0.3460 | 0.2042 | 411 ms | 11.105 GB | 43.8% |
| Query-aware semantic 60% | **0.4516** | **0.3333** | **410 ms** | 11.228 GB | **80.2%** |

The common untrained adapter was measured on the same held-out prompts at step 0: F1 `0.3803`, exact match `0.2250`.

### What this supports

In this internal three-seed pilot, query-aware semantic selection:

- exceeded matched random selection by `+0.1095` mean F1;
- was numerically `+0.0397` mean F1 above full-context fine-tuning;
- retained all annotated supporting paragraphs in 80.2% of training examples, versus 37.5% for matched random selection;
- reduced mean training-step time by about `1.67x` versus full context;
- reduced peak allocated VRAM by about `1.99 GB` versus full context.

The frozen semantic selector added about `5.8 ms` of offline preprocessing per training example. This is reported separately and can be cached in an offline fine-tuning workflow.

### Careful interpretation

This is encouraging evidence of an efficiency/quality trade-off, not a claim that semantic selection is universally better than full context. It is limited to an internal 7B QLoRA pilot, one task family, a small training set, and three seeds. It is not evidence of full-model pretraining savings, 70B behavior, distributed training performance, or production reliability.

## Two operating points

| Mode | Intended use | Current evidence |
| --- | --- | --- |
| `cheap` | Low-overhead generic shortening where no useful task query exists | Efficient, but weaker on dependency-rich evidence retention |
| `semantic` | Query-aware dependency protection when a question/task signal exists | Stronger evidence retention and the best internal MuSiQue QLoRA result |

They are not presented as a winner-take-all replacement. The appropriate policy depends on whether a meaningful task query is available and on the workload.

## Experimental contract

The controlled protocol explicitly reports:

- full context and matched random 60% conditions;
- cheap and query-aware semantic policies at the same budget;
- common step-0 evaluation of the untrained adapter;
- common held-out **full-context** evaluation for every trained adapter;
- peak VRAM, mean training-step time, and selection preprocessing time;
- the rate at which all MuSiQue gold supporting paragraphs were retained.

This separates three possible bottlenecks: a selector fails to retain necessary evidence; evidence is retained but the reader fails to combine it; or generic shortening is enough because a workload is redundant.

## Reproducing the MuSiQue QLoRA protocol

Install the project dependencies plus the QLoRA stack:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install transformers accelerate peft bitsandbytes sentencepiece
```

Download official MuSiQue data separately, then run:

```bash
python3 training/musique_qlora_selection_comparison.py \
  --train-data /path/to/musique_ans_v1.0_train.jsonl \
  --eval-data /path/to/musique_ans_v1.0_dev.jsonl \
  --policies full,random,cheap,semantic \
  --train-cases 96 --eval-cases 80 --steps 192 --batch-size 1 \
  --max-seq-len 4096 --paragraph-tokens 128 --keep-ratio 0.60 \
  --seed 17 --output outputs/musique_qlora_seed17.json
```

Repeat with seeds `29` and `43`. The report includes step-0 quality, policy quality, training evidence coverage, selection overhead, step time, VRAM, and per-example held-out predictions.

## Earlier efficiency signal

An earlier local-text-corpus QLoRA experiment at 7B / 8K, 48 steps, and three seeds found that random and cheap 60% block selection both delivered about `1.67x` step speedup and `6.04 GB` peak VRAM reduction versus full context. Cheap selection had a small held-out-loss advantage over random.

This earlier result is not combined numerically with the MuSiQue task result. It is included only as an efficiency signal showing why dependency-rich task evaluation was needed.

## What is validated today

- Single-GPU 4-bit QLoRA on Qwen2.5 7B
- Training-time physical context selection
- Matched full, random, cheap, and semantic training policies
- Three-seed task-level evaluation on MuSiQue 2-hop QA
- Common full-context held-out F1 and exact-match evaluation
- Training evidence-retention diagnostics
- Step-time, selection-overhead, and peak-VRAM reporting

## What is not validated

- Full-model pretraining cost reduction
- 14B, 32B, 70B, multi-GPU, or distributed training
- Customer data or customer task distributions
- Long-duration convergence
- Production training integration
- Guaranteed quality retention or GPU-cost savings

Any deployment decision requires a controlled validation using the target model, dataset, sequence length, training duration, hardware, and task metric.

## Contributing and feedback

Critical feedback is welcome, especially on evaluation design, selector failure modes, multi-hop evidence retention, and reader-side reasoning after evidence has been retained. Please open an issue with a reproducible case or evaluation suggestion.
