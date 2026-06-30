import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# Importing local modules
from data_pipeline import load_and_merge_data, clean_and_impute
from features import engineer_features


def main():
    print("── 1. LOADING & PREPPING DATA ──")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # CSV paths
    diagnoses_path = os.path.join(base_dir, "data", "diagnoses.csv")
    patients_path = os.path.join(base_dir, "data", "patients.csv")
    outcomes_path = os.path.join(base_dir, "data", "outcomes.csv")

    # Load the data
    df = load_and_merge_data(diagnoses_path, patients_path, outcomes_path)
    admitted = clean_and_impute(df)
    X, y, feature_cols = engineer_features(admitted)

    print("\n── 2. TRAIN / TEST SPLIT ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("\n── 3. DEFINE & FIT MODELS ──")
    # Pipeline for LR
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("Fitting Logistic Regression ...")
    lr.fit(X_train, y_train)
    print("Fitting Random Forest...")
    rf.fit(X_train, y_train)
    print("Fitting XGBoost...")
    xgb.fit(X_train, y_train)

    print("\n── 4. FIT STACKING ENSEMBLE ──")
    meta_lr = LogisticRegression(max_iter=1000, random_state=42)
    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
        ],
        final_estimator=meta_lr,
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    print("\n── 5. SAVE ARTIFACTS ──")
    os.makedirs("models", exist_ok=True)

    BEST_MODEL_NAME = "Stacking (RF + XGB)"
    BEST_MODEL = stacking
    BEST_THRESHOLD = 0.35

    joblib.dump(BEST_THRESHOLD, "../models/best_threshold.pkl")
    joblib.dump(BEST_MODEL, "../models/best_model.pkl")
    joblib.dump(feature_cols, "../models/feature_cols.pkl")

    print(f"Saved best model ({BEST_MODEL_NAME}) and threshold ({BEST_THRESHOLD}) to /models")


if __name__ == "__main__":
    main()