# INTELIPS: Intelligent Email Priority System

**Author:** Karim Semaan
**Course:** CS6120 - Natural Language Processing
**Semester:** Fall 2025

## Overview

INTELIPS is an email prioritization system built on the Enron corpus. An LLM (served via the Groq API) auto-labels ~25,640 emails into three priority levels, then five architectures are trained and benchmarked against each other: a TF-IDF + metadata XGBoost baseline, a context-aware MLP, a BERT + attention model, fine-tuned BERT, and PAEPS, an exploratory personalization network with per-user embeddings.

**Key Results (real LLM-annotated data):**
- **72.85% macro-F1** with a weighted ensemble, the best real-data result
- **72.18% macro-F1** XGBoost baseline (the spread across all five architectures is small)
- **25,640 emails** annotated using the Groq API for ~$10
- A separate personalization experiment on synthetic personas reached 90.91% F1; that figure is **not comparable** to the numbers above (see [Earlier synthetic-label experiment](#earlier-synthetic-label-experiment-paeps-personalization))

## Research Question

*Can we build an email prioritization system that learns personalized importance patterns for different users?*

**Honest takeaway:** on real LLM-annotated labels, model architecture barely matters (72.18% to 72.85% macro-F1 from a feature baseline to a weighted ensemble). The binding constraint is label quality, not network capacity. Personalization looked dramatic only on a synthetic persona set, which is exactly why that result is quarantined in the limitations section below.

## Repository Structure

```
INTELIPS_SUBMISSION/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── smart_annotation_pipeline.py   # Groq API annotation pipeline
├── train_final_model.py           # PAEPS model training script
├── run_personalized_simple.py     # Personalized model implementation
├── test_custom_models.py          # Baseline model experiments
├── data/
│   ├── raw/
│   │   └── enron_emails.csv        # Original Enron email dataset (517k emails need to download from kaggle)
│   └── enron_annotated_1k.csv     # Initial annotated dataset (1k emails)
│   └── enron_annotated_5k.csv   # Intermediate annotated dataset (5k emails)
│   └── enron_annotated_25k.csv  # Final cleaned dataset
├── notebooks/
│   ├── 1_data_exploration.ipynb       # Dataset exploration
│   ├── 2_annotation_pipeline.ipynb    # LLM annotation process
│   ├── 3_baseline_models.ipynb        # Baseline experiments
│   ├── 4_context_aware_models.ipynb   # MLP & attention models
│   ├── 5_final_improved_models.ipynb  # BERT & HCEC models
│   └── 6_personalized_email_priority_system.ipynb  # PAEPS model
└── results/
    ├── personalized_model_results.json  # Final model metrics
    ├── all_experiments.json             # Experiment logs
    ├── logs/                            # Individual experiment logs
    └── figures/presentation/            # All visualizations
```

## Models Implemented

Benchmarked on the real LLM-annotated Enron data (macro-F1):

| Model | F1 Score | Description |
|-------|----------|-------------|
| **Weighted Ensemble (best)** | **72.85%** | Weighted combination of the models below |
| XGBoost Baseline | 72.18% | TF-IDF + metadata features |
| HCEC (Attention) | 71.20% | BERT + multi-head attention |
| Context-Aware MLP | 69.36% | Separate text/context branches |
| BERT Fine-tuned | 64.87% | BERT-base-uncased |

PAEPS (personalized user embeddings) is excluded from this table: its 90.91% F1 was measured on a synthetic persona set, not on the real annotated data, so the numbers are not comparable. See [Earlier synthetic-label experiment](#earlier-synthetic-label-experiment-paeps-personalization).

## Exploratory Personalization: User Embeddings

We model 4 user personas with different priority patterns:

| User Type | Priority Focus |
|-----------|----------------|
| CEO | Board meetings, investor communications, strategy |
| Developer | Bug reports, production issues, code reviews |
| Manager | Deadlines, milestones, project blockers |
| Sales | Customer communications, deals, proposals |

Each user receives a 20-dimensional embedding that modifies how the model weighs email features. Note: these personas and their labels are synthetically constructed (no real per-user ground truth exists in Enron), so all PAEPS numbers belong to the synthetic-label experiment described below.

## Dataset

- **Source:** Enron Email Corpus (517,000 emails)
- **Annotated:** 25,640 emails using Groq API
- **Cost:** ~$10 total
- **Labels:** 3-level priority (Low, Normal, Critical)

| Priority | Count | Percentage |
|----------|-------|------------|
| Low | 14,478 | 56.5% |
| Normal | 9,373 | 36.6% |
| Critical | 1,789 | 7.0% |

## Features (20 total)

1. **User Personalization (25%)** - User embeddings, role-specific keywords
2. **Temporal Features (22%)** - Hour, day, business hours, deadlines
3. **Sender Intelligence (18%)** - Sender importance, relationship strength
4. **Content Analysis (15%)** - Urgency keywords, sentiment
5. **Email Context (14%)** - Reply/forward, recipients, attachments

## Earlier synthetic-label experiment (PAEPS personalization)

An earlier version of this README headlined **90.91% F1** (and 99.07% on high-confidence predictions at 88.8% coverage, a "+25.96% improvement over baselines"). Those numbers are real outputs of the PAEPS experiment, but it is important to be precise about what they measured:

- The Enron corpus has no per-user priority ground truth, so the personalization experiment **synthetically constructed** four user personas with persona-specific priority rules, and built its dataset from generated synthetic emails plus real emails relabeled under those rules.
- PAEPS was then trained and evaluated **on that synthetic persona set**. The 90.91% therefore measures how well the network recovers the constructed persona labeling scheme, not how well it prioritizes email for real users.
- Because the evaluation labels come from the same persona construction that drives the personalization features, the score is inflated relative to any real-world setting and **cannot be compared** to the 72.18% to 72.85% macro-F1 results on the real LLM-annotated data above.

The experiment still has value as a proof of concept: it shows the user-embedding architecture can learn strongly user-conditioned priority functions when such signal exists. Validating that on real, human-labeled, per-user data is the obvious next step, alongside auditing the LLM annotations themselves with a human-labeled gold set.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Annotation Pipeline
```bash
python smart_annotation_pipeline.py
```

### Train PAEPS Model
```bash
python train_final_model.py
```

### Run Personalized Model
```bash
python run_personalized_simple.py
```

## Results Visualization

All figures are in `results/figures/presentation/`:
- Model comparison charts
- Feature importance analysis
- High-confidence prediction breakdown
- Data scaling analysis

## License

MIT License
