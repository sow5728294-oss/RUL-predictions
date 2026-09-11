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

#load model
model = joblib.load(model_path)

#load raw and processed test data
test_data_processed = pd.read_csv(test_path)
test_data_raw = pd.read_csv(
    raw_test_path,
    sep=r"\s+",
    header=None
)

#load RUL to calculate ground truth for test set
RUL_data = pd.read_csv(
    RUL_path,
    sep=r"\s+",
    header=None
)

#set column name for raw test data
test_data_raw = test_data_raw.rename(columns={
    0: "unit",
    1: "cycle"
})

#calculate the cycle until the last cycle for that trajectories in test data (Test data stop randomly before failure)
max_cycle = test_data_raw.groupby("unit")["cycle"].transform("max")
test_data_processed['RUL_not_until_failure'] = max_cycle - test_data_raw['cycle']

#create a mapping to fetch how many cycle left after the data suddenly stop for each unit trajectories
rul_mapping = pd.Series(
    RUL_data[0].values,
    index=range(1, len(RUL_data) + 1)
)

#create a column which contain how many cycle left after the data stop for that specific unit
test_data_processed["final_RUL"] = test_data_raw["unit"].map(rul_mapping)
#calculate the ACTUAL AMOUNT OF CYCLE LEFT UNTIL FAILURE for each row
test_data_processed["actual_RUL"] = (
    test_data_processed["RUL_not_until_failure"]
    + test_data_processed["final_RUL"]
)

#set the actual RUL as y for testing
y_test = test_data_processed["actual_RUL"]

#get all the name of model's features
feature_names = model.get_booster().feature_names

# Select those features from test data
x_test = test_data_processed[feature_names]

#use model to make prediction on test set
y_pred = model.predict(x_test)


#compare y_pred and y_test to evaluate performance of model
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

#print MAE, RMSE and r2 Score
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

