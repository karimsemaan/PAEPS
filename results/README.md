# Results — metric provenance map

Every headline number quoted for INTELIPS/PAEPS traces to a committed artifact in this
repository. This file maps each metric to its source so claims are independently verifiable.

| Metric | Value | Source artifact |
| --- | --- | --- |
| XGBoost baseline (TF-IDF + metadata), macro-F1 | **72.18%** | `results/ALL_MODELS_RESULTS.json` (`f1_macro: 0.7218`) and `FINAL_REPORT.pdf` §4.2 |
| Weighted soft-voting ensemble (HCEC 0.30 + MLP 0.70), macro-F1 | **72.85%** | `results/ensemble_results.json` — extracted verbatim from the executed output of `notebooks/5_final_improved_models.ipynb`, section 8 (Ensemble Voting) |
| Ensemble accuracy / precision / recall | 79.00% / 71.58% / 74.48% | Same notebook cell output; mirrored in `results/ensemble_results.json` |
| Context-Aware MLP, macro-F1 | 69.36% | `FINAL_REPORT.pdf` results table |
| HCEC (BERT + attention), macro-F1 | 71.20% | `FINAL_REPORT.pdf` results table |
| Fine-tuned BERT, macro-F1 | 64.87% | `results/ALL_MODELS_RESULTS.json` (`f1_macro: 0.6487…`) |
| PAEPS personalized model, F1 | 90.91% | `results/personalized_model_results.json` — **synthetic-persona evaluation only.** This is an exploratory proof-of-concept number, *not* comparable to the real-data metrics above and never presented as one. |
| Dataset size | ~25,640 Enron emails | `data/enron_annotated_25k.csv` (+1k/5k subsets) |

## Why two "best" numbers?

On the **real LLM-annotated data**, the classical XGBoost baseline (72.18%) proved very hard
to beat; the weighted ensemble edges it at 72.85%. The personalized PAEPS network's 90.91%
was measured on **synthetic personas** built to demonstrate the personalization mechanism,
which is why it is quarantined from the real-data comparison everywhere this project is
described.
