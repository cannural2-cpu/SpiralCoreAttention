# SpiralCoreAttention

## Context efficiency for long-context open-weight LLMs

SpiralCoreAttention is an experimental AI infrastructure project for reducing unnecessary context before LLM inference.

The goal is simple: when a model receives a long document, RAG output, or oversized prompt, SpiralCoreAttention selects the most relevant context instead of passing the complete input directly to the model.

This can reduce prompt-processing load, KV-cache memory pressure, and long-context inference latency.

---

## Why this matters

Long-context LLM workloads often include:

- Large documents
- Document-heavy RAG pipelines
- Repeated retrieved chunks
- Noisy knowledge bases
- Oversized prompts
- Irrelevant context

Passing all available context into a model can increase GPU memory usage, latency, and inference cost.

SpiralCoreAttention is designed to evaluate whether a smaller, relevant context can preserve useful information while lowering inference overhead.

---

## How it works

```text
Raw long-context input
        ↓
Anchor-based context selection
        ↓
Relevant evidence and reduced context
        ↓
Underlying language model
```

SpiralCoreAttention does not replace or retrain the underlying LLM.

It acts as a context-selection layer before inference.

---

## Internal validation

Current internal testing was performed with:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: NVIDIA GeForce RTX 5090
- Input size: approximately 17K tokens
- Test set: 10 synthetic long-context evaluation cases

### Internal benchmark signal

| Metric | Baseline | Spiral |
|---|---:|---:|
| p95 latency | 1677.67 ms | 1269.83 ms |
| Mean peak VRAM | 8.836 GB | 6.466 GB |
| Spiral fallback count | - | 0 / 10 |

In this internal test, Spiral showed:

- Approximately 24% lower p95 latency
- Approximately 27% lower peak VRAM usage
- No fallback events across the 10 tested cases

These results are early internal measurements only.

They are not a production guarantee and may change depending on model, hardware, context length, prompt distribution, quantisation, serving stack, and output requirements.

---

## Quality evaluation

Efficiency alone is not enough.

Every Spiral evaluation includes side-by-side baseline and Spiral output review.

Current observations:

- Stronger results on document-heavy and structured context tasks
- Mixed results on reasoning-heavy or fact-sensitive tasks
- Quality retention remains an active area of development

No performance result should be used without output-quality review on the target workload.

---

## Current status

SpiralCoreAttention is currently an experimental prototype.

Validated internally:

- Single-GPU inference
- Qwen2.5-1.5B-Instruct
- Synthetic long-context prompts
- Latency and peak-VRAM comparison
- Baseline-versus-Spiral reporting

Not yet validated:

- 70B models
- vLLM integration
- Multi-GPU inference
- Production traffic
- Customer-specific enterprise workloads

---

## Pilot evaluation

SpiralCoreAttention is looking for a small number of technical design partners running long-context open-weight LLM workloads.

The pilot is designed as an isolated benchmark inside the partner's own environment.

Pilot scope:

- 7-day technical validation
- 10-30 anonymised or synthetic representative prompts
- Baseline-versus-Spiral comparison
- p95 latency measurement
- Peak VRAM measurement
- Output comparison
- Fallback reporting
- No production access required
- No raw customer-data export required

The purpose of the pilot is to measure real impact on a specific workload before discussing production integration.

---

## Roadmap

- Improve quality retention
- Improve fallback behaviour
- Add reproducible benchmark configurations
- Validate on larger open-weight models
- Add vLLM support
- Validate multi-GPU inference
- Test enterprise long-document and RAG workloads
- Build production-ready integration options

---

## Contact

Can Yilmaz Nural  
Founder, SpiralCoreAttention  

Email: cannural.contact@gmail.com

---

## Disclaimer

SpiralCoreAttention is experimental research software.

All benchmark figures in this repository are internal results from a specific model, GPU, configuration, and synthetic test set. Results may vary materially across different models, workloads, hardware, and inference frameworks.
