import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def engineer_features(admitted):
    # ── 6. ENCODE CATEGORICALS ────────────────────────────────
    cat_cols = [
        "sex", "smoking_status", "alcohol_use",
        "exercise_level", "insurance_type",
        "discharge_disposition",
    ]
    le = LabelEncoder()
    for col in cat_cols:
        if admitted[col].isna().any():
            admitted[col] = admitted[col].fillna("Unknown")
        admitted[col + "_enc"] = le.fit_transform(admitted[col].astype(str))

    # ── 1. DISEASE BURDEN ─────────────────────────────────────
    dx_cols = [c for c in admitted.columns if c.startswith("dx_")]
    admitted["num_comorbidities"] = admitted[dx_cols].sum(axis=1)

    # ── 2. ABNORMAL VITALS SCORE ──────────────────────────────────────────────
    admitted["high_systolic"]  = (admitted["systolic_bp"]   > 140).astype(int)
    admitted["high_diastolic"] = (admitted["diastolic_bp"]  > 90).astype(int)
    admitted["high_hr"]        = (admitted["heart_rate"]    > 100).astype(int)
    admitted["low_hr"]         = (admitted["heart_rate"]    < 60).astype(int)
    admitted["fever"]          = (admitted["temperature_f"] > 100.4).astype(int)
    admitted["hypothermia"]    = (admitted["temperature_f"] < 96.8).astype(int)
    admitted["obese_flag"]     = (admitted["bmi"]           >= 30).astype(int)
    admitted["underweight_flag"] = (admitted["bmi"]         < 18.5).astype(int)

    vital_flags = [
        "high_systolic", "high_diastolic", "high_hr", "low_hr",
        "fever", "hypothermia", "obese_flag", "underweight_flag",
    ]
    admitted["abnormal_vitals_score"] = admitted[vital_flags].sum(axis=1)

    # ── 3. BMI CATEGORIES ─────────────────────────────────────────────────────
    bmi_bins   = [0, 18.5, 25, 30, 35, np.inf]
    bmi_labels = ["underweight", "normal", "overweight", "obese", "severely_obese"]
    admitted["bmi_category"] = pd.cut(admitted["bmi"], bins=bmi_bins, labels=bmi_labels, right=False)
    admitted["bmi_category_enc"] = pd.Categorical(admitted["bmi_category"], categories=bmi_labels, ordered=True).codes

    # ── 4. AGE GROUPS ─────────────────────────────────────────────────────────
    age_bins   = [0, 40, 60, 75, np.inf]
    age_labels = ["18-40", "40-60", "60-75", "75+"]
    admitted["age_group"] = pd.cut(admitted["age"], bins=age_bins, labels=age_labels, right=False)
    admitted["age_group_enc"] = pd.Categorical(admitted["age_group"], categories=age_labels, ordered=True).codes

    # ── 5. HOSPITAL SEVERITY FEATURES ─────────────────────────────────────────
    los_bins   = [0, 3, 7, 14, np.inf]
    los_labels = ["short", "medium", "long", "extended"]
    admitted["los_category"] = pd.cut(admitted["length_of_stay_days"], bins=los_bins, labels=los_labels, right=False)
    admitted["los_category_enc"] = pd.Categorical(admitted["los_category"], categories=los_labels, ordered=True).codes

    admitted["icu_admission"] = admitted["icu_admission"].fillna(0).astype(int)

    charlson_bins   = [0, 1, 3, 5, np.inf]
    charlson_labels = ["low", "moderate", "high", "very_high"]
    admitted["charlson_tier"] = pd.cut(admitted["charlson_index"], bins=charlson_bins, labels=charlson_labels, right=False)
    admitted["charlson_tier_enc"] = pd.Categorical(admitted["charlson_tier"], categories=charlson_labels, ordered=True).codes

    # ── 6. INTERACTION FEATURES ───────────────────────────────────────────────
    admitted["icu_x_los"]            = admitted["icu_admission"] * admitted["length_of_stay_days"]
    admitted["charlson_x_los"]       = admitted["charlson_index"] * admitted["length_of_stay_days"]
    admitted["comorbidities_x_age"]  = admitted["num_comorbidities"] * admitted["age"]
    admitted["hf_x_ckd"]            = (admitted["dx_heart_failure"] * admitted["dx_chronic_kidney_disease"])

    # ── 7. PULSE PRESSURE ─────────────────────────────────────────────────────
    admitted["pulse_pressure"] = admitted["systolic_bp"] - admitted["diastolic_bp"]

    # ── 9. FINAL FEATURE LIST ─────────────────────────────────────────────────
    feature_cols = (
        ["age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
         "temperature_f", "charlson_index", "length_of_stay_days",
         "icu_admission", "icu_days", "total_charges_usd", "primary_drg",
         "pulse_pressure"]
        + dx_cols
        + ["num_comorbidities", "abnormal_vitals_score",
           "bmi_category_enc", "age_group_enc",
           "los_category_enc", "charlson_tier_enc",
           "icu_x_los", "charlson_x_los",
           "comorbidities_x_age", "hf_x_ckd"]
        + vital_flags
        + ["sex_enc", "smoking_status_enc", "alcohol_use_enc",
           "exercise_level_enc", "insurance_type_enc",
           "discharge_disposition_enc"]
    )

    feature_cols = [c for c in feature_cols if c in admitted.columns]

    X = admitted[feature_cols].copy()
    y = admitted["readmitted_30d"].copy()

    return X, y, feature_cols