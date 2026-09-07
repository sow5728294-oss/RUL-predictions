import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from xgboost import XGBRegressor 
from sklearn.model_selection import RandomizedSearchCV

BASE_DIR = Path(__file__).resolve().parent.parent

processed_dir = BASE_DIR / "Data" / "target data" / "processed"

train_data = pd.read_csv(
    processed_dir / "train_processed.csv"
)

val_data = pd.read_csv(
    processed_dir / "val_processed.csv"
)

x_train = train_data.drop(columns=["unit","RUL"])
y_train = train_data["RUL"]

x_val = val_data.drop(columns=["unit","RUL"])
y_val = val_data["RUL"]

def test_param():

    model_arch = XGBRegressor(
        random_state = 42,
        n_jobs = -1
    )

    param_list = {
        "n_estimators":[100,200,300,400,500,600,700],
        "learning_rate":[0.01,0.05,0.1,0.2,0.3],
        "max_depth":[3,4,5,6,7,8,9,10],
        "subsample":[0.5,0.7,1.0],
        "colsample_bytree":[0.5,0.7,1.0],
    }
    
    search_model = RandomizedSearchCV(
        estimator = model_arch,
        param_distributions = param_list,
        n_iter = 50,
        cv = 5,
        scoring = "neg_mean_absolute_error",
        random_state = 42,
        n_jobs = -1
    )

    search_model.fit(x_train,y_train)

    print(search_model.best_params_)


model = XGBRegressor(
    n_estimators = 500,
    learning_rate = 0.01,
    max_depth = 9,
    random_state = 42,
    subsample = 0.7,
    colsample_bytree = 0.7,

)

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

train_mae = mean_absolute_error(y_train,y_train_pred)
val_mae = mean_absolute_error(y_val,y_val_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train,y_train_pred)
)
val_rmse = np.sqrt(
    mean_squared_error(y_val,y_val_pred)
)

train_r2 = r2_score(y_train,y_train_pred)
val_r2 = r2_score(y_val,y_val_pred)

print(f"FOR TRAIN DATA: \n MAE: {train_mae} \n RMSE: {train_rmse} \n R2: {train_r2}")
print(f"FOR VALIDATION DATA: \n MAE: {val_mae} \n RMSE: {val_rmse} \n R2: {val_r2}")


