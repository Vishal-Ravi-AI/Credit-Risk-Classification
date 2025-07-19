
#  Credit Risk Classification Using Classical and Open-Source AI Models

##  Files Included
- `Classical Models.ipynb` – Contains all steps from preprocessing to evaluation using classical machine learning and deep learning models.
- `Open_Source_Models.ipynb` – Fine-tunes open-source LLMs (LLaMA 3.1 and OpenChat 3.5) for credit risk classification using LoRA.
- `credit_alpaca.jsonl` – Dataset formatted for fine-tuning open-source models.
- `credit_prompts.txt` – Raw structured prompts used for zero-shot LLM classification.
- `README.md` – Project documentation.
- `*.joblib` – Serialized models for inference.

##  Project Overview
This project focuses on classifying whether a customer applying for an insurance quote is a **good or bad credit risk** (`1` = good, `0` = bad). It compares classical ML models, deep learning, and fine-tuned LLMs for both performance and deployment feasibility.

##  Workflow Summary

###  Classical Models
- **Data Cleaning**: Removed missing values, normalized labels.
- **Feature Engineering**:
  - Binary mapping (`True`/`False` → 1/0)
  - Extracted year, month, weekday from timestamp.
  - Speed binning and one-hot encoding for categorical features.
- **Model Training**:
  - Trained multiple models: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM, and XGBoost.
  - Hyperparameter tuning via GridSearchCV and cross-validation.
  - A shallow deep learning model was implemented using **TensorFlow/Keras**.
- **Evaluation**:
  - Accuracy, classification report, confusion matrix, and cost-based analysis.
  - **XGBoost** performed the best with **~92% accuracy** on test data.

###  Open-Source AI Models
- **LLaMA 3.1-8B Instruct** and **OpenChat 3.5-1210** were fine-tuned using LoRA adapters with `trl`'s `SFTTrainer`.
- Used structured prompts (zero-shot) and small custom dataset (`credit_alpaca.jsonl`) for instruction tuning.
- Final predictions were generated using Hugging Face's `pipeline()` API.
- Results were compared with classical models to evaluate feasibility.

##  Tools & Libraries
- scikit-learn (XGBoost, SVM, RF, LR, etc.)
- TensorFlow / Keras
- Hugging Face Transformers
- `peft`, `trl`, LoRA
- Huggingface Hub, joblib

##  Results
- **XGBoost** outperformed all models with an accuracy >91%.
- Classical models were significantly faster and easier to interpret.
- Open-source LLMs demonstrated flexibility for inference but required heavy compute and longer response times.
- LoRA tuning was effective in adapting general-purpose LLMs for structured data classification.

##  Notes
- PyTorch was **not used** in this implementation.
- Mistral was **not included** in the final version.
- All code is designed for readability and reproducibility.
