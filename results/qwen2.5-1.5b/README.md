# Results — Qwen2.5-1.5B-Instruct

All files of the 1.5B study. File names carry `1.5b` like the other sizes; the name each notebook
originally wrote on Kaggle is given in brackets.

| Step | File | What it is |
|---|---|---|
| 00 pilot | `00_pilot_1.5b-fp16_gsm8k.jsonl`, `00_pilot_1.5b-fp16_triviaqa.jsonl` | 50-question FP16 pilot per task (`pilot_*_fp16.jsonl`) |
| 02 quantize | `02_quantize_1.5b_results.json`, `02_quantize_1.5b_settings.json` | OPT-125M reproduction of the GPTQ paper + first Qwen models (plain GPTQ, C4 and math calibration) (`02_results.json`, `02_settings.json`) |
| 02b check | `02b_gptq_check_1.5b_results.json` | Perplexity of GPTQ with act-order (the version used), plus a layer-level check (`02b_results.json`) |
| 01 answers | `gen_1.5b-{fp16,rtn4,gptq4}_{gsm8k,triviaqa}.jsonl`, `gen_1.5b-*_summary.json` | Every answer of every model (`gen_<model>_<task>.jsonl`). TriviaQA has 17,944 rows: the dataset lists some questions twice; notebook 03 keeps the first copy (9,960 questions). |
| 03 flips | `03_results_1.5b.json`, `03_harmful_flips_1.5b.jsonl`, `03_per_question_1.5b.csv` | Accuracy, flips, McNemar, overlap, AUROC, length (lenient TriviaQA scoring) (`03_*.json/jsonl/csv`) |
| 04 judge 1 | `04_judged_1.5b_judge1.jsonl`, `04_labels_1.5b_judge1.jsonl`, `04_results_1.5b_judge1.json` | First judge, Gemini 3.5 Flash Lite, both tasks. **Its facts labels are the final facts labels**; its math labels were replaced by judge 2 (`04_judged.jsonl`, `04_labels.jsonl`, `04_results.json`) |
| 04 judge 2 | `04_judged_1.5b_math_v2.jsonl`, `04_labels_1.5b_math_v2.jsonl`, `04_results_1.5b_math_v2.json` | Final math judge, gpt-oss-120b (Groq), sharper definitions (`04_*_math_v2.*`) |
| 04 agreement | `04_hand_sample_1.5b.csv`, `04_hand_sample_key_1.5b.csv`, `04_hand_labels_1.5b.csv` | Blind 70-flip sample, its answer key, and the labels of a second LLM annotator (Claude) used for Cohen's κ (`04_hand_*.csv`) |

Final labels: facts → `04_labels_1.5b_judge1.jsonl` (task `triviaqa`); math → `04_labels_1.5b_math_v2.jsonl`.

To rerun notebooks 04/05 from these files, rename them back to the bracketed Kaggle names.
