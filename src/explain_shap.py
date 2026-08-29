"""
Explain the XGBoost model with SHAP - global feature importance plus
per-customer explanations for a couple of example high-risk accounts.

Requires train_model.py to have been run first (to loads its saved artifacts).

Run standalone:
    python src/explain_shap.py
"""
# %%
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_SPOT = Path(__file__).parent.parent / "outputs"


def load_artifacts():
    model = joblib.load(OUTPUT_SPOT / "xgb_model.joblib")
    X_test = pd.read_csv(OUTPUT_SPOT / "X_test.csv")
    return model, X_test


def global_importance(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    #bar chart: mean |SHAP value| per feature, ranked
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_SPOT / "shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_SPOT / 'shap_global_importance.png'}")

    #beeswarm: shows direction + magnitude, not just rank
    fig2 = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_SPOT / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {OUTPUT_SPOT / 'shap_beeswarm.png'}")

    return shap_values


def top_feature_ranking(shap_values, X_test, top_n=10):
    """Print a plain-text ranked list"""
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    print(f"\nTop {top_n} features by mean |SHAP value|:")
    print(importance.head(top_n).to_string(index=False))
    return importance


def explain_one_customer(model, X_test, row_idx=0):
    """Per-customer explanation, which factors push THIS account's risk up/down."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[row_idx], show=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_SPOT / f"shap_customer_{row_idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_SPOT / f'shap_customer_{row_idx}.png'}")


if __name__ == "__main__":
    model, X_test = load_artifacts()
    shap_values = global_importance(model, X_test)
    top_feature_ranking(shap_values, X_test)
    explain_one_customer(model, X_test, row_idx=0)
