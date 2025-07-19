# Credit Risk Classification Using Classical and Open-Source AI Models

## Files Included

- `Classical Models.ipynb` – End-to-end pipeline using classical ML and deep learning models.
- `Open_Source_Models.ipynb` – Fine-tunes open-source LLMs (LLaMA 3.1 and OpenChat 3.5) for credit risk classification.
- `Cargo_final_timestamp_speed_encoded.xlsx` – Cleaned and preprocessed dataset used for classical model training.
- `scored_full_dataset.xlsx` – Model-scored output dataset used for evaluation and analysis.
- `credit_alpaca.jsonl` – Custom dataset formatted for instruction-tuning LLMs.
- `credit_prompts.txt` – Structured prompts used in zero-shot LLM inference.
- `xgb_credit_risk_pipeline_final.pkl` – Serialized XGBoost pipeline model for inference.
- `batch_score.py` – Script to load the serialized model and generate predictions on new data.
- `README.md` – Project documentation.

---

## Project Overview

This project classifies insurance applicants as either good or bad credit risks (1 = good, 0 = bad). It compares classical machine learning, deep learning, and fine-tuned open-source LLMs to evaluate accuracy, efficiency, and feasibility in production scenarios.

---

## Workflow Summary

### Classical Models

- **Data Cleaning**: Removed missing values, normalized labels.
- **Feature Engineering**:
  - Binary mapping (True/False → 1/0)
  - Extracted year, month, weekday from timestamp.
  - Speed binning and one-hot encoding for categorical features.
- **Model Training**:
  - Trained models: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM, XGBoost.
  - Hyperparameter tuning via `GridSearchCV` and cross-validation.
  - Shallow deep learning model with TensorFlow/Keras.
- **Evaluation**:
  - Metrics: Accuracy, classification report, confusion matrix, cost-based analysis.
  - ✅ **XGBoost** performed best (~92% test accuracy).
- **Deployment**:
  - Serialized using `joblib` (saved as `xgb_credit_risk_pipeline_final.pkl`).
  - Scoring script: `batch_score.py`.

### Open-Source AI Models

- **LLMs**: LLaMA 3.1–8B Instruct and OpenChat 3.5–1210.
- **Fine-tuning**:
  - LoRA adapters using `trl.SFTTrainer` with `credit_alpaca.jsonl`.
- **Zero-shot Inference**:
  - Prompt-based classification using `credit_prompts.txt`.
  - Hugging Face `pipeline()` for generation.
- **Evaluation**:
  - Compared LLMs to classical models in accuracy and runtime.

---

## Tools & Libraries

- `scikit-learn`, `XGBoost`, `SVM`, `LogisticRegression`
- `TensorFlow` / `Keras`
- `transformers`, `trl`, `peft` (LoRA)
- `joblib` (model serialization)
- `pandas`, `openpyxl`, `Hugging Face Hub`

---

## Results

- **XGBoost** achieved ~92% accuracy and best overall performance.
- Classical models were fast, interpretable, and production-friendly.
- Fine-tuned LLMs were flexible but required higher compute and longer inference time.

---

## Notes

- `.pkl` file (`xgb_credit_risk_pipeline_final.pkl`) stores the serialized pipeline model.
- Prompt tuning was done using a small JSONL dataset, not full tabular data.
- Mistral was not included in the final version.
- All notebooks are designed for reproducibility and learning.

