# INTELIPS: Intelligent Email Priority System

**Author:** Karim Semaan
**Course:** CS6120 - Natural Language Processing
**Semester:** Fall 2025

## Overview

INTELIPS is a personalized email prioritization system that learns user-specific importance patterns. Unlike traditional approaches that treat all users identically, INTELIPS uses user embeddings to understand that the same email may have different priorities for different people.

**Key Results:**
- **90.91% F1 Score** with personalized model (PAEPS)
- **99.07% F1 Score** on high-confidence predictions (88.8% coverage)
- **+25.96% improvement** over baseline models
- **25,640 emails** annotated using Groq API for ~$10

## Research Question

*Can we build an email prioritization system that learns personalized importance patterns for different users?*

**Key Insight:** A simple model that understands WHO is reading the email dramatically outperforms complex models that only understand WHAT is written.

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

| Model | F1 Score | Description |
|-------|----------|-------------|
| XGBoost Baseline | 72.18% | TF-IDF + metadata features |
| Context-Aware MLP | 69.36% | Separate text/context branches |
| HCEC (Attention) | 71.20% | BERT + multi-head attention |
| BERT Fine-tuned | 64.87% | BERT-base-uncased |
| **PAEPS (Ours)** | **90.91%** | Personalized user embeddings |

## Key Innovation: User Embeddings

We model 4 user personas with different priority patterns:

| User Type | Priority Focus |
|-----------|----------------|
| CEO | Board meetings, investor communications, strategy |
| Developer | Bug reports, production issues, code reviews |
| Manager | Deadlines, milestones, project blockers |
| Sales | Customer communications, deals, proposals |

Each user receives a 20-dimensional embedding that modifies how the model weighs email features.

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
