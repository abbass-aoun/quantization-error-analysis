# Does Smarter Quantization Change the *Kind* of Mistakes LLMs Make?

Final project for **COE 549 – Advanced LLMs** (Lebanese American University, Fall 2026, Dr. Samer Saab Jr.) by **Abbass Aoun**.

Quantization barely changes a model's *accuracy*, yet many individual answers still change ("flips").
This project compares **GPTQ** with **round-to-nearest (RTN)** on the **same 4-bit grid** and asks not only
*how many* answers break, but *how* they break.

| | |
|---|---|
| **Target paper** | Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023 |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Tasks** | GSM8K (math reasoning, 1,319 test questions) · TriviaQA (factual recall, 17,944 validation questions, no context) |
| **Models compared** | FP16 · RTN-4bit · GPTQ-4bit (C4 calibration) · GPTQ-4bit (math calibration); all 4-bit, group size 128, same grid |
| **Compute** | Kaggle, one free NVIDIA T4 |

## Research questions

- **RQ1:** How many answers flip after RTN vs GPTQ quantization? (McNemar test)
- **RQ2:** Do RTN and GPTQ produce different *error types* in the harmful flips? (χ² test; LLM judge checked against a hand-labelled sample with Cohen's κ)
- **RQ3:** Does the original model's confidence predict which answers flip? (AUROC)
- **Extra:** Does calibrating GPTQ on math text help math but hurt factual recall?

**Harmful flip:** the FP16 model is right and the quantized model is wrong. **Repair flip:** the reverse.

## Repository layout

```
notebooks/
  00_pilot_fp16.ipynb    first small FP16 test (50 + 50 questions); kept for the record
  01_generate.ipynb      runs one model on all GSM8K + TriviaQA questions (set TAG), with checkpointing
  02_quantize.ipynb      RTN and GPTQ from scratch: OPT-125M reproduction + Qwen RTN / GPTQ-C4 / GPTQ-math
  02b_gptq_check.ipynb   diagnostic: why plain GPTQ lost to RTN on Qwen; act-order + static groups
src/
  quantlib.py            Quantizer, RTN, GPTQ (Algorithm 1: Cholesky, lazy blocks, damping,
                         act-order, static groups, true-sequential)
  helpers.py             answer extraction and scoring for GSM8K and TriviaQA
results/README.md        all numbers obtained so far
demo/rtn-vs-gptq-simulator.html   interactive RTN vs GPTQ demo (open in any browser)
docs/                    literature survey (PDF + LaTeX) and presentation (PDFs, speaker notes, HTML deck)
```

The notebooks are **self-contained**: the code from `src/` is copied into them so they run on Kaggle without extra files.
`src/` holds the same code as readable modules.

## How to run (Kaggle)

1. **02_quantize.ipynb**, then **02b_gptq_check.ipynb**. Each produces quantized model folders in its Output.
2. **01_generate.ipynb** once per model: set `TAG` to `"fp16"`, `"rtn4"`, `"gptq4"` or `"gptq4math"` in cell 2.
   Attach the outputs of 02 / 02b as Inputs (**+ Add Input → Your Work**).
3. Always use **Save Version → Save & Run All (Commit)** with **GPU T4 x2** and **Internet On**, so outputs are kept.
   Runs save as they go; a crashed run resumes if its output is attached as an Input.

## Status

- [x] GPTQ paper reproduced on OPT-125M (within 0.1 perplexity of Table 3)
- [x] FP16 baseline on all 19,263 questions
- [x] RTN and GPTQ models built; GPTQ with act-order + static groups beats RTN on WikiText2 and C4
- [ ] RTN and GPTQ exams (01)
- [ ] Flip analysis (03) · LLM judge + κ (04)
- [ ] Final paper

## Notes

- Quantization is *simulated* (weights rounded to the 4-bit grid and stored in 16-bit). This gives the same answers as real INT4 kernels but says nothing about speed.
- Large files (model folders, per-question `.jsonl` answers) stay on Kaggle and are not in this repository.
- AI tools were used to help write code and documents; all experiments were run and checked by the author.
