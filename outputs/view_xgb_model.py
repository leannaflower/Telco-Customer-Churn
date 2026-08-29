"""
This file is to view the xgb_model.joblib file
"""
import joblib
from pathlib import Path

data = joblib.load(Path(__file__).parent / "xgb_model.joblib")

print(type(data))
print(data)