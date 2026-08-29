"""
This file, eda.py, is exploratory analysis, looking at churn rate by segment, and saved as plots in outputs/
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_loader import load_clean

# %%
OUTPUT_SPOT = Path(__file__).parent.parent / "outputs"
OUTPUT_SPOT.mkdir(exist_ok=True)

def churn_rate_by(df, column, ax=None):
    """ bar chart of churn rate grouped by a categorical column """
    rates = df.groupby(column)["Churn"].mean().sort_values(ascending=False)
    ax = rates.plot(kind='bar', ax=ax, color='#4C72B0')
    ax.set_ylabel("Churn Rate")
    ax.set_title(f"Churn rate by {column}")
    ax.axhline(df["Churn"].mean(), color="gray", linestyle="--", linewidth=1)
    return ax

def run_eda(df):
    #the categorical driver for the dataset
    segments = ["Contract", "PaymentMethod", "InternetService", "TechSupport"]

    fig, axes = plt.subplots(2, 2, figsize=(12,9))
    for ax, col in zip(axes.flat, segments):
        churn_rate_by(df, col, ax=ax)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUT_SPOT / "churn_by_segment.png", dpi=150)
    print(f"Saved {OUTPUT_SPOT / 'churn_by_segment.png'}")

    #tenure distribution split by churn
    fig2, ax2 = plt.subplots(figsize=(7,5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack", ax=ax2)
    ax2.set_title("Tenure distrubtion by churn status")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_SPOT / "tenure_by_churn.png", dpi=150)
    print(f"Saved {OUTPUT_SPOT / 'tenure_by_churn.png'}")

    #quick printed summary of the numbers
    print("\n--- Churn Rate by Segment (Numeric) ---")
    for col in segments:
        print(f"\n{col}:")
        print(df.groupby(col)["Churn"].mean().sort_values(ascending=False))

if __name__ == "__main__":
    df = load_clean()
    run_eda(df)