"""
Feature engineering: derived features + categorical encoding.

Run standalone:
    python src/feature_engineering.py
"""
# %%
import pandas as pd

from data_loader import load_clean


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of the raw columns."""
    df = df.copy()

    # Tenure buckets — churn risk often concentrates in the first year
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
    )

    # Ratio of total spend to monthly rate — low ratio flags newer/at-risk
    # accounts even within the same tenure bucket.
    df["charge_ratio"] = df["TotalCharges"] / df["MonthlyCharges"].replace(0, 1)

    # Count of subscribed add-on services — bundle depth is a classic
    # stickiness signal.
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["addon_count"] = (df[addon_cols] == "Yes").sum(axis=1)

    # Flag for the highest-risk payment method identified in EDA
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    return df


def encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categoricals; leave target untouched."""
    df = df.copy()
    target = df["Churn"]
    features = df.drop(columns=["Churn"])

    features = pd.get_dummies(features, drop_first=True)
    features["Churn"] = target
    return features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return encode_for_model(add_derived_features(df))


if __name__ == "__main__":
    df = load_clean()
    featured = build_features(df)
    print(f"Feature matrix: {featured.shape[0]} rows, {featured.shape[1] - 1} features")
    print(featured.columns.tolist())
