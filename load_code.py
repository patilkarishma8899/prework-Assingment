"""
Financial Risk Analysis for Loan Approval
==========================================
Author: Karishma Patil
Role: ML Engineer | 47Billion, Bengaluru

This module implements the end-to-end ML pipeline for loan approval prediction.
It includes preprocessing, dual-model training (classifier + regressor),
and a critical inference function used in the Flask API.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
APPROVAL_THRESHOLD = 0.5
NUMERIC_COLS = [
    'age', 'annual_income', 'credit_score',
    'employment_years', 'existing_debt', 'loan_term_months'
]
CATEGORICAL_COLS = ['employment_type', 'property_ownership', 'loan_purpose']
TARGET_CLASS = 'loan_approved'
TARGET_REG = 'loan_amount'

# Global objects (loaded at Flask startup)
knn_imputer = None
scaler = None
clf_model = None
reg_model = None
iqr_bounds = {}


# ─────────────────────────────────────────────
# STEP 1: LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw loan application data from CSV."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Class distribution:\n{df[TARGET_CLASS].value_counts()}")
    return df


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────
def compute_iqr_bounds(df: pd.DataFrame) -> dict:
    """
    Compute IQR-based outlier bounds on the training set.
    These bounds are saved and reused at inference time.
    """
    bounds = {}
    for col in NUMERIC_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = {
            'lower': Q1 - 1.5 * IQR,
            'upper': Q3 + 1.5 * IQR
        }
    return bounds


def clip_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Apply IQR-based outlier clipping using pre-computed bounds."""
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = df[col].clip(
            lower=bounds[col]['lower'],
            upper=bounds[col]['upper']
        )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def build_preprocessing_pipeline(X_train: pd.DataFrame):
    """
    Fit KNN imputer and StandardScaler on training data only.
    Returns fitted objects to be saved and reused at inference.
    """
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_imputed)

    return imputer, sc, X_scaled


# ─────────────────────────────────────────────
# STEP 3: TRAIN MODELS
# ─────────────────────────────────────────────
def train_classifier(X_train, y_train):
    """
    Train Random Forest Classifier with GridSearchCV hyperparameter tuning.
    SMOTE is applied to training data only to handle class imbalance.
    """
    # Apply SMOTE only on training set — avoid data leakage
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    print(f"Best classifier params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def train_regressor(X_train, y_train):
    """
    Train Random Forest Regressor for loan amount estimation.
    Only trained on approved loan records.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best regressor params: {grid_search.best_params_}")
    return grid_search.best_estimator_


# ─────────────────────────────────────────────
# STEP 4: EVALUATE MODELS
# ─────────────────────────────────────────────
def evaluate_classifier(model, X_test, y_test):
    """Print classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Classifier Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred))


def evaluate_regressor(model, X_test, y_test):
    """Print regression metrics."""
    y_pred = model.predict(X_test)

    print("=== Regressor Evaluation ===")
    print(f"R² Score:  {r2_score(y_test, y_pred):.4f}")
    print(f"MAE:       {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE:      {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")


# ─────────────────────────────────────────────
# STEP 5: CRITICAL INFERENCE FUNCTION
# ─────────────────────────────────────────────
def preprocess_and_predict(raw_input: dict) -> dict:
    """
    ══════════════════════════════════════════════════════════════
    CRITICAL FUNCTION — used in Flask API for every prediction call
    ══════════════════════════════════════════════════════════════

    End-to-end preprocessing and dual-model prediction for a
    single loan application submitted via the REST API.

    Parameters:
        raw_input (dict): Raw customer application data from API request.
            Expected keys: age, annual_income, credit_score,
                           employment_years, existing_debt,
                           loan_term_months, employment_type,
                           property_ownership, loan_purpose

    Returns:
        dict: {
            "approved": bool,
            "approval_probability": float,
            "estimated_loan_amount": float or None
        }

    Flow:
        1. Convert dict → DataFrame
        2. Encode categoricals
        3. Impute missing values using pre-fitted KNNImputer
        4. Clip outliers using pre-computed IQR bounds
        5. Scale using pre-fitted StandardScaler
        6. Classify: approved / rejected
        7. If approved → estimate loan amount
        8. Return structured response
    """
    # Step 1: Convert to DataFrame for pipeline compatibility
    df = pd.DataFrame([raw_input])

    # Step 2: Encode categorical fields
    df = encode_categoricals(df)

    # Align columns with training schema (handle unseen categories)
    expected_cols = knn_imputer.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Step 3: Impute missing values using KNN (k=5, fitted on training data)
    df_imputed = pd.DataFrame(
        knn_imputer.transform(df),
        columns=df.columns
    )

    # Step 4: Clip outliers using IQR bounds computed from training set
    for col in NUMERIC_COLS:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].clip(
                lower=iqr_bounds[col]['lower'],
                upper=iqr_bounds[col]['upper']
            )

    # Step 5: Scale features using pre-fitted StandardScaler
    df_scaled = scaler.transform(df_imputed)

    # Step 6: Classify — is the loan approved?
    approval_proba = clf_model.predict_proba(df_scaled)[0][1]
    approval_decision = int(approval_proba >= APPROVAL_THRESHOLD)

    # Step 7: Regress — estimate loan amount only if approved
    loan_amount = None
    if approval_decision == 1:
        loan_amount = float(reg_model.predict(df_scaled)[0])

    # Step 8: Return structured prediction
    return {
        "approved": bool(approval_decision),
        "approval_probability": round(float(approval_proba), 4),
        "estimated_loan_amount": round(loan_amount, 2) if loan_amount else None
    }


# ─────────────────────────────────────────────
# STEP 6: SAVE / LOAD MODELS
# ─────────────────────────────────────────────
def save_artifacts(imputer, sc, clf, reg, bounds, path="models/"):
    """Serialize all model artifacts to disk."""
    import os
    os.makedirs(path, exist_ok=True)
    joblib.dump(imputer, f"{path}knn_imputer.pkl")
    joblib.dump(sc, f"{path}scaler.pkl")
    joblib.dump(clf, f"{path}clf_model.pkl")
    joblib.dump(reg, f"{path}reg_model.pkl")
    joblib.dump(bounds, f"{path}iqr_bounds.pkl")
    print(f"All artifacts saved to {path}")


def load_artifacts(path="models/"):
    """Load all saved artifacts into global variables (called once at Flask startup)."""
    global knn_imputer, scaler, clf_model, reg_model, iqr_bounds
    knn_imputer = joblib.load(f"{path}knn_imputer.pkl")
    scaler = joblib.load(f"{path}scaler.pkl")
    clf_model = joblib.load(f"{path}clf_model.pkl")
    reg_model = joblib.load(f"{path}reg_model.pkl")
    iqr_bounds = joblib.load(f"{path}iqr_bounds.pkl")
    print("All model artifacts loaded successfully.")


# ─────────────────────────────────────────────
# TRAINING PIPELINE (run once)
# ─────────────────────────────────────────────
def run_training_pipeline(filepath: str):
    """Full training pipeline — run offline to train and save models."""
    df = load_data(filepath)

    # Separate features and targets
    X = df.drop(columns=[TARGET_CLASS, TARGET_REG])
    y_class = df[TARGET_CLASS]
    y_reg = df[TARGET_REG]

    # Encode categoricals
    X = encode_categoricals(X)

    # Train-test split FIRST — then preprocess
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # Compute IQR bounds on training set only
    bounds = compute_iqr_bounds(X_train)

    # Clip outliers
    X_train = clip_outliers(X_train, bounds)
    X_test = clip_outliers(X_test, bounds)

    # Fit imputer and scaler on training set only
    imputer, sc, X_train_scaled = build_preprocessing_pipeline(X_train)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    X_test_scaled = sc.transform(X_test_imputed)

    # Train models
    clf = train_classifier(X_train_scaled, y_class_train)
    reg = train_regressor(X_train_scaled, y_reg_train)

    # Evaluate
    evaluate_classifier(clf, X_test_scaled, y_class_test)
    evaluate_regressor(reg, X_test_scaled, y_reg_test)

    # Save
    save_artifacts(imputer, sc, clf, reg, bounds)


if __name__ == "__main__":
    # Example usage:
    # run_training_pipeline("data/loan_applications.csv")

    # Example inference (after loading artifacts):
    # load_artifacts()
    # result = preprocess_and_predict({
    #     "age": 35,
    #     "annual_income": 750000,
    #     "credit_score": 720,
    #     "employment_years": 5,
    #     "existing_debt": 50000,
    #     "loan_term_months": 36,
    #     "employment_type": "salaried",
    #     "property_ownership": "owned",
    #     "loan_purpose": "home"
    # })
    # print(result)
    pass