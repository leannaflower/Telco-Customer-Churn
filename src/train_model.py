"""
Train a baseline logistic regression and a stronger XGBoost model.
Evaluates with ROC-AUC, precision/recall, and precision-at-top-decile
(the metric that matters most for a real retention campaign, since you can
only realistically target a limited number of accounts).

Run standalone:
    python src/train_model.py
"""
# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

from data_loader import load_clean
from feature_engineering import build_features

MODEL_SPOT = Path(__file__).parent.parent / "outputs"
MODEL_SPOT.mkdir(exist_ok=True)


def precision_at_k(y_true, y_score, k=0.2): #precision = a performance metric that measure how many of the positive predictions made by a model were actually correct
    """Precision among the top-k% highest-scored accounts

    Simulates a real retention campaign: ranking every customer by predicted
    churn risk, then only have budget to contact the riskiest slice...."of the top k% 
    the model flags, what fraction actually churned?"
    """
    n = int(len(y_score) * k)                       # how many accounts the top-k% slice holds
    top_idx = np.argsort(y_score)[::-1][:n]         # indices of the n highest risk scores
    # mean of a 0/1 churn column over that slice == the precision we want
    return y_true.iloc[top_idx].mean() if hasattr(y_true, "iloc") else y_true[top_idx].mean()


def split_data(featured: pd.DataFrame):
    """Separate the label from the features and carve out a held-out test set"""
    X = featured.drop(columns=["Churn"])            # everything except the target (Churn)
    y = featured["Churn"]                           # the 0/1 target
    # stratify = y keeps the ~27%   churn rate identical in train and test;
    # I am choosing random_state = 42 (consistant within this project - making the split reproducible run to run
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def train_baseline(X_train, y_train):
    """interpretable reference model to beat: logistic regression"""
    # Logistic regression is scale-sensitive, so standardize first. The scaler is fit on TRAIN ONLY and returned so the same transform can be reused on test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")  # class_weight="balanced" up-weights the minority (churn) class so the model doesn't just learn to predict "no churn" for everyone
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_xgb(X_train, y_train):
    """Stronger gradient-boosted trees model"""
    # scale_pos_weight handles the class imbalance (~27% churn rate): ratio of negatives to positives tells XGBoost how much to weight churners
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,          # number of boosting rounds (trees)
        max_depth=4,               # shallow trees -> less overfitting
        learning_rate=0.05,        # small steps; pairs with the higher n_estimators
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
    )
    # Trees don't need scaling, so raw X_train goes straight in.
    model.fit(X_train, y_train)
    return model


def evaluate(name, y_test, y_score):
    """Print the three metrics we care about and return them as a row dict"""
    # ROC-AUC: ranking quality across all thresholds (1.0 perfect, 0.5 coin flip).
    auc = roc_auc_score(y_test, y_score)
    # Precision in the top 20% riskiest -- the campaign-relevant number.
    p_at_20 = precision_at_k(y_test, y_score, k=0.2)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Precision @ top 20% riskiest: {p_at_20:.3f} "
          f"(vs. base rate {y_test.mean():.3f})")     # base rate = churn rate if you picked at random
    # classification_report: precision/recall/F1 at a fixed 0.5 probability cutoff.
    print(classification_report(y_test, (y_score > 0.5).astype(int)))
    return {"model": name, "roc_auc": auc, "precision_at_20": p_at_20}


def main():
    # --- 1. Data -> features ---
    df = load_clean()                               # raw CSV, cleaned (see data_loader.py)
    featured = build_features(df)                   # + derived features + one-hot encoding
    X_train, X_test, y_train, y_test = split_data(featured)

    results = []                                    # one metrics row per model

    # --- 2. Baseline: logistic regression ---
    lr_model, scaler = train_baseline(X_train, y_train)
    # reuse the train-fitted scaler on test; [:, 1] = probability of churn (class 1)
    lr_score = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    results.append(evaluate("Logistic Regression", y_test, lr_score))

    # --- 3. Stronger model: XGBoost ---
    xgb_model = train_xgb(X_train, y_train)
    xgb_score = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_score))

    # --- 4. Persist everything explain_shap.py and main.py need ---
    joblib.dump(xgb_model, MODEL_SPOT / "xgb_model.joblib")   # the trained model
    X_test.to_csv(MODEL_SPOT / "X_test.csv", index=False)     # held-out features for SHAP / eval
    y_test.to_csv(MODEL_SPOT / "y_test.csv", index=False)     # held-out labels
    X_train.to_csv(MODEL_SPOT / "X_train.csv", index=False)   # SHAP background/reference set

    pd.DataFrame(results).to_csv(MODEL_SPOT / "model_comparison.csv", index=False)
    print(f"\nSaved model + eval artifacts to {MODEL_SPOT}/")


if __name__ == "__main__":
    main()
