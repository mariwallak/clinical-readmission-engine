# 30-Day Hospital Readmission Prediction

## 1. Executive Summary
This repository contains a predictive analytics pipeline engineered to forecast 30-day patient readmission risks. Designed to mitigate institutional healthcare penalties and optimize patient care transitions, the system integrates a decoupled data pipeline, a stacked ensemble machine learning architecture, and localized Explainable AI (XAI) diagnostics.

## 2. System Architecture and Separation of Concerns
The project enforces a strict separation of concerns across data ingestion, feature engineering, model serialization, and visualization layers.

```text
readmission-risk-predictor/            
├── data/
│   └── *.csv                # Source clinical datasets (Git-ignored)
├── models/
│   ├── best_model.pkl       # Serialized Stacking Classifier pipeline binary
│   ├── best_threshold.pkl   # Precision-Recall optimized decision threshold (0.35)
│   └── feature_cols.pkl     # Preserved deterministic feature schema
├── notebooks/
│   ├── 01_eda.ipynb         # Statistical profile exploration & distribution analysis
│   └── 02_shap_explainability.ipynb # Global attribution & calibration diagnostics
├── src/
│   ├── __init__.py          # Module initialization root
│   ├── data_pipeline.py     # Deterministic merge and imputation engine
│   ├── features.py          # Vectorized feature transformations & encoding
│   └── train.py             # Model training, validation, and serialization
├── .gitignore               # Excludes virtual environments and protected data
└── requirements.txt         # Version-locked dependency manifesto
```

### 2.1 Core Subsystems
* **Data Pipeline (`src/data_pipeline.py`):** Handles multi-source data ingestion (demographics, administrative metrics, and diagnostics). Implements memory-optimized joining logic to eliminate index-alignment blowout on high-volume datasets.
* **Feature Engineering (`src/features.py`):** Computes domain-specific interaction terms (e.g., ICU utilization crossed with length of stay, comorbidity burdens stratified by age groups) and scales continuous vectors cleanly.
* **Training Pipeline (`src/train.py`):** Trains and serializes the machine learning stack, utilizing dynamic relative paths to compute the project root across heterogeneous execution environments.

## 3. Machine Learning Architecture
To handle complex non-linear clinical relationships while maintaining robust classification boundaries, the system deploys a Stacked Ensemble Classifier:

* **Base Estimators:** A robust Random Forest Classifier paired with an optimized Extreme Gradient Boosting (XGBoost) model.
* **Meta-Learner:** A Logistic Regression model that learns optimal blending weights from the base estimators' prediction probability matrices.
* **Class Imbalance Optimization:** Imbalance in readmission event rates is mitigated at the loss-function level using targeted class penalisations and custom positive scaling weights (`scale_pos_weight`), maximizing the Area Under the Precision-Recall Curve (AUPRC).

## 4. Operational Requirements and Installation

### 4.1 Prerequisites
The codebase is validated for Python 3.11. If executing on Apple Silicon architecture, the OpenMP runtime framework must be present to support accelerated C-extensions (XGBoost). Run the following command in your terminal:

brew install libomp

### 4.2 Installation and Environment Setup
Clone the repository and isolate dependencies using a local virtual environment. Run these commands line-by-line in your terminal:
```bash
git clone https://github.com/mariwallak/clinical-readmission-engine.git
cd clinical-readmission-engine
python3.11 -m venv .venv1
source .venv1/bin/activate
pip install -r requirements.txt
```

### 4.3 Data Directory Configuration
Populate the local data directory with the following CSV source artifacts prior to execution:
* data/patients.csv
* data/diagnoses.csv
* data/outcomes.csv

## 5. Execution Protocol

### 5.1 Pipeline Execution and Model Serialization
To execute the automated pipeline end-to-end—including data cleaning, feature extraction, ensemble training, and artifact serialization—run the training script from the root terminal:

python src/train.py

This evaluates the model stack and generates three immutable binaries within the root /models folder: best_model.pkl, best_threshold.pkl, and feature_cols.pkl.

## 6. Interpretability and Decision Optimization

### 6.1 Classification Boundary Calibration
The pipeline rejects standard 0.5 classification thresholds in favor of an empirically optimized decision boundary of 0.35. This operational setting was determined via cost-benefit coordinate grids designed to minimize false negatives (critical overlooked high-risk patients) while preserving resource-efficient specificity constraints.

### 6.2 Explainable AI Framework
To ensure clinical accountability, the architecture rejects pure black-box inference by integrating Tree SHAP (SHapley Additive exPlanations). Localized prediction instances are decomposed via local attribution scales (Waterfall plots), illustrating the exact directional forces and magnitude vectors driving an individual patient's score. Global feature distributions are verified via beeswarm plots to ensure alignment with established clinical literature.