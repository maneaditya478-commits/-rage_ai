# TokenRage: Studying AI Reply Token Usage vs. User Frustration

TokenRage is an end-to-end machine learning and data science investigation examining whether verbose, repetitive, or long AI responses lead to increased user frustration and anger in subsequent conversational turns.

## Status
Work in progress. Phase 1 and a first version of Phase 2 are complete. Label quality fixes are planned before statistical analysis.

## Research Questions
1. **Token Usage vs. User Anger**: Does higher token count in an AI reply correlate with user anger or frustration in the *subsequent* user message?
2. **Behavioral Triggers**: Which specific patterns correlate most strongly with frustration: verbose/lengthy replies, repetitive content (n-gram overlap), apologies loops, refusal/hedging, instruction mismatch, or hallucinations?
3. **Pre-Emptive Risk Prediction**: Can a predictive model estimate the probability that an AI reply will trigger a frustrated reaction *before* the user sends their next message?

## Repository Structure
```
├── data/
│   ├── raw/                 # Downloaded raw dataset shards (WildChat)
│   └── processed/           # Processed conversation transitions & feature tables
├── notebooks/               # Jupyter/EDA notebooks
├── src/
│   ├── data/                # Data download, loading, PII scrubbing, turn pairing
│   ├── features/            # Token counting (tiktoken), n-gram repetition, apology counts
│   ├── labeling/            # Emotion scoring (RoBERTa), heuristic rules, validation
│   ├── models/              # Logistic regression, gradient boosting (XGBoost/LightGBM)
│   └── utils/               # Constants, configurations, random seeds
├── reports/
│   └── figures/             # Generated statistical figures and charts
├── scripts/                 # Reproducible phase execution scripts
├── tests/                   # Pytest unit tests for features, data loader, and labels
├── requirements.txt         # Project dependencies
└── README.md
```

## Reproducibility
- Single-command phase execution via `scripts/run_phase1.py`, etc.
- Fixed random seeds (`seed=42`) across data splits and models.
- Unit testing with `pytest`.
