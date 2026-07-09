# Player Churn Prediction Dashboard
### MSc Capstone Project (In Progress — August 2026)

## Overview
An end-to-end churn-risk prediction pipeline built on player behavioural engagement 
data, deployed as an interactive Streamlit dashboard for retention decision support.

**Dataset:** 40,034 player records · 13 variables · 25.8% churn-risk class  
**Source:** [Online Gaming Behavior Dataset — Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

## Problem
Player-level longitudinal data is structurally inaccessible through industry channels. 
This study responds by building a fully reproducible open-data pipeline, using 
behavioural engagement snapshots as a proxy for churn risk in the absence of session 
timestamps or inactivity records.

## Models Compared

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.619 | 0.842 | 0.713 | 0.904 |
| Random Forest | 0.917 | 0.860 | 0.887 | 0.935 |
| **Gradient Boosting** | **0.925** | **0.865** | **0.894** | **0.938** |

All models evaluated on a held-out test set (8,007 records) at default threshold 
t = 0.50. 5-fold stratified cross-validation confirmed results without overfitting.

Gradient Boosting selected for its lowest F1 variance across folds (SD = 0.0022) 
and strongest PR-AUC (0.8878 vs no-skill baseline of 0.2578).

## Key Findings

**Session behaviour dominates all other predictors**  
Permutation importance analysis on the held-out test set identified SessionsPerWeek 
(PR-AUC drop = 0.462) and AvgSessionDurationMinutes (PR-AUC drop = 0.306) as the 
two dominant predictors. Combined, they account for nearly all discriminative power. 
Demographic, contextual, and monetisation features contribute negligibly once 
session-based features are present.

**Non-linearity explains the precision gap**  
The 0.306 precision gap between Gradient Boosting and Logistic Regression reflects 
a fundamental mismatch between a linear decision boundary and a non-linear churn 
signal. A player with low session frequency but high session duration presents a 
different risk profile than one low on both — an interaction Logistic Regression 
cannot capture without explicit feature engineering.

**Threshold selection is a business decision**  
F1-maximising threshold analysis on out-of-fold training predictions identified 
t = 0.42 as optimal. Relative to t = 0.50, this gains 36 additional true positives 
at the cost of 52 additional false positives. Whether this trade-off is beneficial 
depends on the operational cost of retention interventions in the deploying 
organisation. The dashboard makes this trade-off interactive.

**Predicted probabilities are reliable at the extremes**  
Brier Score of 0.0525 (27.4% of no-skill baseline) confirms probabilities are 
meaningful as risk estimates. The bimodal distribution means most players score near 
0 or 1, so the model behaves closer to a binary classifier than a continuous risk 
scorer in practice. Raw scores in the mid-range should be interpreted with caution.

## Dashboard Features
- **Model Evaluation** — adjustable risk threshold with real-time confusion matrix, 
ROC-AUC, PR-AUC, and threshold comparison panel (t = 0.42 vs t = 0.50)
- **Risk Explorer** — segment-level risk ranking with Top-N and borderline case 
filters, column selection, colour-coded flags, and CSV export
- **What-if Scoring** — single-player churn probability with custom progress bar 
showing predicted score relative to current threshold position

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tools
Python · pandas · scikit-learn · imbalanced-learn · matplotlib · Streamlit

## Files
- `app.py` — Streamlit dashboard
- `cp_churn_analysis.ipynb` — model training, threshold analysis, and evaluation
- `cp_churn_eda.ipynb` — exploratory data analysis
- `cp_churn_threshold_viz.ipynb` — threshold visualisation
