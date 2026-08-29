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

# the proportion of retrieved items in the top-K list that are genuinely relevant to the user
def precision_at_k(y_true, y_score, k=0.2):
    """Precision among the top-k% highest-scored accounts."""
    n = int(len(y_score) * k)
    top_idx = np.argsort(y_score)[::-1][:n]
    return y_true.iloc[top_idx].mean() if hasattr(y_true, "iloc") else y_true[top_idx].mean()


def split_data(featured: pd.DataFrame):
    X = featured.drop(columns=["Churn"])
    y = featured["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def train_baseline(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_xgb(X_train, y_train):
    # scale_pos_weight handles the class imbalance (~27% churn rate)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(name, y_test, y_score):
    auc = roc_auc_score(y_test, y_score)
    p_at_20 = precision_at_k(y_test, y_score, k=0.2)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Precision @ top 20% riskiest: {p_at_20:.3f} "
          f"(vs. base rate {y_test.mean():.3f})")
    print(classification_report(y_test, (y_score > 0.5).astype(int)))
    return {"model": name, "roc_auc": auc, "precision_at_20": p_at_20}


def main():
    df = load_clean()
    featured = build_features(df)
    X_train, X_test, y_train, y_test = split_data(featured)

    results = []

    #baseline
    lr_model, scaler = train_baseline(X_train, y_train)
    lr_score = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    results.append(evaluate("Logistic Regression", y_test, lr_score))

    #stronger model
    xgb_model = train_xgb(X_train, y_train)
    xgb_score = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_score))

    #persist everything explain_shap.py and main.py need
    joblib.dump(xgb_model, MODEL_SPOT / "xgb_model.joblib")
    X_test.to_csv(MODEL_SPOT / "X_test.csv", index=False)
    y_test.to_csv(MODEL_SPOT / "y_test.csv", index=False)
    X_train.to_csv(MODEL_SPOT / "X_train.csv", index=False)

    pd.DataFrame(results).to_csv(MODEL_SPOT / "model_comparison.csv", index=False)
    print(f"\nSaved model + eval artifacts to {MODEL_SPOT}/")


if __name__ == "__main__":
    main()
