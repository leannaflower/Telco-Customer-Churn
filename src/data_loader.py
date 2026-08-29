import pandas as pd
from pathlib import Path

# %%
DATA_PATH = Path(__file__).parent.parent / "Data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. Download the Telco Churn file please. and place it in the Data folder (See README)."
        )
    return pd.read_csv(path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # clean the db
    # found "TotalCharges" as "Object" -> change to float64
        # blank TotalCharges are all tenure == 0 (brand-new customers, not yet billed)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Changing "Churn" to be 0 or 1 (Yes/No)
    df['Churn'] = (df['Churn'] == "Yes").astype(int)

    # customerID is not a feature, so drop it
    df = df.drop(columns=["customerID"])
    return df

def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))

if __name__ == "__main__":
    df = load_clean()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    print(df.dtypes)