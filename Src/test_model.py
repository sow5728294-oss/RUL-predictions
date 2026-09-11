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

model = joblib.load(model_path)
test_data_processed = pd.read_csv(test_path)
RUL_data = pd.read_csv(
    RUL_path,
    sep=r"\s+",
    header=None
)
test_data_raw = pd.read_csv(
    raw_test_path,
    sep=r"\s+",
    header=None
)

test_data_raw = test_data_raw.rename(columns={
    0: "unit",
    1: "cycle"
})

max_cycle = test_data_raw.groupby("unit")["cycle"].transform("max")
test_data_processed['RUL_not_until_failure'] = max_cycle - test_data_raw['cycle']

rul_mapping = pd.Series(
    RUL_data[0].values,
    index=range(1, len(RUL_data) + 1)
)
test_data_processed["final_RUL"] = test_data_raw["unit"].map(rul_mapping)
test_data_processed["actual_RUL"] = (
    test_data_processed["RUL_not_until_failure"]
    + test_data_processed["final_RUL"]
)

y_test = test_data_processed["actual_RUL"]

feature_names = model.get_booster().feature_names

# Select those features from test data
x_test = test_data_processed[feature_names]

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== TEST RESULTS ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

