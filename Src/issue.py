from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "Model" / "xgb_model.pkl"
test_path = BASE_DIR / "Data" / "target data" / "processed" / "test_processed.csv"
RUL_path = BASE_DIR / "Data" / "target data" / "raw" / "RUL_FD002.txt"
raw_test_path = BASE_DIR / "Data" / "target data" / "raw" / "test_FD002.txt"


train_data = pd.read_csv(
    BASE_DIR / "Data" / "target data" / "processed" / "train_processed.csv"
)

test_data = pd.read_csv(
    BASE_DIR / "Data" / "target data" / "processed" / "test_processed.csv"
)

model = joblib.load(model_path)

feature_names = model.get_booster().feature_names


print("TRAIN FEATURE RANGE")
print(train_data[feature_names].describe().T[["mean", "std", "min", "max"]])

print("\nTEST FEATURE RANGE")
print(test_data[feature_names].describe().T[["mean", "std", "min", "max"]])
