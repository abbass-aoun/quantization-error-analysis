# Results so far

## 1. GPTQ reproduction: OPT-125M, 4-bit per-row, WikiText2 perplexity (notebook 02, Part A)

| model | GPTQ paper (Table 3) | ours |
|---|---|---|
| FP16 | 27.65 | 27.66 |
| RTN-4 | 37.28 | 37.29 |
| GPTQ-4 | 31.12 | 31.22 |

## 2. FP16 baseline: Qwen2.5-1.5B-Instruct (notebook 01, `TAG = "fp16"`)

| task | questions | accuracy | notes |
|---|---|---|---|
| GSM8K | 1,319 | **75.5%** (996) | 99.0% gave a `\boxed{}` answer; 5 hit the 1,024-token cap |
| TriviaQA (no context) | 17,944 | **35.2%** (alias "contains" match) | 30.2% exact match |

Greedy decoding, no repetition penalty, identical prompts for every model.

## 3. Qwen2.5-1.5B quantized to 4 bits, group size 128: perplexity (notebooks 02 and 02b)

| model | WikiText2 | C4 |
|---|---|---|
| FP16 | 9.649 | 15.647 |
| RTN-4 | 10.738 | 17.229 |
| GPTQ-4, plain (notebook 02) | 11.182 | 16.643 |
| **GPTQ-4, act-order + static groups** | **10.186** | **16.284** |
| GPTQ-4, act-order + static groups + true-sequential | 10.261 | 16.283 |
| GPTQ-4 math calibration, plain (notebook 02) | 12.45 | – |

All quantized models verified to use at most 16 distinct values per (row, group).
**Chosen GPTQ setting: act-order + static groups** (beats RTN on both datasets; static groups keep RTN's exact grid).

## 4. Layer-level diagnostic: GPTQ output error ÷ RTN output error (notebook 02b, check 2)

Below 1 means GPTQ is better. "C4" = inputs like the calibration text; "wiki" = unseen WikiText2 inputs.

| layer | plain · C4 | plain · wiki | act · C4 | act · wiki |
|---|---|---|---|---|
| 1.self_attn.q_proj | 0.348 | 0.457 | 0.277 | 0.377 |
| 1.self_attn.o_proj | 0.090 | 0.154 | 0.068 | 0.119 |
| 1.mlp.down_proj | 0.020 | 0.154 | 0.004 | 0.009 |
| 5.self_attn.q_proj | 0.364 | 0.534 | 0.284 | 0.459 |
| 5.self_attn.o_proj | 0.330 | 0.556 | 0.295 | 0.507 |
| 5.mlp.down_proj | 0.579 | **1.091** | 0.410 | **1.216** |
| 14.self_attn.q_proj | 0.411 | 0.553 | 0.304 | 0.446 |
| 14.self_attn.o_proj | 0.309 | 0.347 | 0.239 | 0.306 |
| 14.mlp.down_proj | 0.580 | 0.841 | 0.529 | 0.820 |
| 27.self_attn.q_proj | 0.523 | 0.815 | 0.396 | 0.717 |
| 27.self_attn.o_proj | 0.134 | 0.247 | 0.108 | 0.216 |
| 27.mlp.down_proj | 0.094 | 0.284 | 0.058 | 0.251 |

Reading: plain GPTQ fits its calibration text closely (every ratio is higher on wiki than on C4); act-order lowers the ratio in 23 of 24 cases.

## 5. Flips and error types

*Pending: needs the RTN and GPTQ runs of notebook 01.*
