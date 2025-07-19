import joblib
import pandas as pd

# 1. Load your model and full dataset
pipeline = joblib.load("xgb_credit_risk_pipeline_final.pkl")
df_full  = pd.read_excel("Cargo_final_timestamp_speed_encoded.xlsx")

# 2. Score all rows at once
features = df_full.drop("allow", axis=1)
df_full["predicted_class"]     = pipeline.predict(features)
df_full["approve_probability"] = pipeline.predict_proba(features)[:, 1]

# 3. Save the scored file
df_full.to_excel("scored_full_dataset.xlsx", index=False)
print(f"Scored {len(df_full)} rows → scored_full_dataset.xlsx")
