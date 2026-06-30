import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def load_and_merge_data(diagnoses_path, patients_path, outcomes_path):
    # ── 1. LOAD ───────────────────────────────────────────────────────────────
    diagnoses_csv = pd.read_csv(diagnoses_path, index_col=0)
    patients_csv = pd.read_csv(patients_path, index_col=0)
    outcomes_csv = pd.read_csv(outcomes_path, index_col=0)

    # Merging based on patient ID
    df = patients_csv.join(diagnoses_csv, how="left", rsuffix='_diag') \
        .join(outcomes_csv, how="left", rsuffix='_out')

    print(f"Shape after merge: {df.shape}")
    return df


def clean_and_impute(df):
    # ── 3. FILTER TO ADMITTED PATIENTS ────────────────────────────────────────
    admitted = df[df["readmitted_30d"].notna()].copy()
    print(f"Admitted patients: {len(admitted)}")
    print(f"Readmission rate:  {admitted['readmitted_30d'].mean():.1%}")

    # ── 4. HANDLE DATES ───────────────────────────────────────────────────────
    for col in ["admission_date", "discharge_date", "visit_date"]:
        if col in admitted.columns:
            admitted[col] = pd.to_datetime(admitted[col], errors="coerce")

    # ── 5. IMPUTE NUMERICS ────────────────────────────────────────────────────
    numeric_cols = [
        "age", "bmi", "systolic_bp", "diastolic_bp",
        "heart_rate", "temperature_f", "charlson_index",
    ]
    for col in numeric_cols:
        median = admitted[col].median()
        n_missing = admitted[col].isna().sum()
        admitted[col] = admitted[col].fillna(median)
        if n_missing:
            print(f"  Imputed {n_missing} missing in '{col}' with median={median:.1f}")

    # days_to_readmission: only meaningful for readmitted=1 patients; fill 0 otherwise
    admitted["days_to_readmission"] = admitted["days_to_readmission"].fillna(0)

    # icu_days: if icu_admission==0, icu_days should be 0
    admitted["icu_days"] = admitted["icu_days"].fillna(0)

    return admitted