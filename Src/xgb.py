import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from xgboost import XGBRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold

#create path to processed data
BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "Data" / "target data" / "processed"


#get train data and validation data
train_data = pd.read_csv(
    processed_dir / "train_processed.csv"
)

val_data = pd.read_csv(
    processed_dir / "val_processed.csv"
)

#get x and y for train data set
x_train = train_data.drop(columns=["unit","RUL"])
y_train = train_data["RUL"]

#get x and y for validation data set
x_val = val_data.drop(columns=["unit","RUL"])
y_val = val_data["RUL"]


#function to test best hyperparameter combination
def test_param():

    model_arch = XGBRegressor(
        random_state = 42,
        n_jobs = -1
    )

    param_list = {
        "n_estimators":[200,400,600,800,1000],
        "learning_rate":[0.01,0.05,0.1,0.2,0.3],
        "max_depth":[3,4,5,6],
        "subsample":[0.5,0.7,1.0],
        "colsample_bytree":[0.5,0.7,1.0],
    }
    
    cv = GroupKFold(n_splits=5)


    search_model = RandomizedSearchCV(
        estimator = model_arch,
        param_distributions = param_list,
        n_iter = 30,
        cv = cv,
        scoring = "neg_mean_absolute_error",
        random_state = 42,
        n_jobs = -1,
        verbose = 2
    )

    search_model.fit(
        x_train,
        y_train,
        groups = train_data["unit"]
        )

    print(search_model.best_params_)


model = XGBRegressor(
    n_estimators = 500, # no need further reduce, already no overfittinh
    learning_rate = 0.01, #increase wont help
    max_depth =6, #increase wont help 
    random_state = 42,
    subsample = 0.7,
    colsample_bytree = 0.7,
    )

#create log y for train data
y_train_log = np.log1p(y_train)

#train the model with x_train and log y train
model.fit(x_train,y_train_log)

#get prediction for x_train and x_val (the output is logged)
y_train_pred_log = model.predict(x_train)
y_val_pred_log = model.predict(x_val)

#"unlog" the prediction to get original scale of y
y_train_pred = np.expm1(y_train_pred_log)
y_val_pred = np.expm1(y_val_pred_log)


#compare prediction y with actual y and calculate error 
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


#print error score for MAE, RMSE and r2
print(f"FOR TRAIN DATA: \n MAE: {train_mae} \n RMSE: {train_rmse} \n R2: {train_r2}")
print(f"FOR VALIDATION DATA: \n MAE: {val_mae} \n RMSE: {val_rmse} \n R2: {val_r2}")


#display amount of error in each RUL range
def display_error_by_RUL_range(y_val,y_val_pred):
    result = pd.DataFrame({
        "actual":y_val,
        "predicted":y_val_pred
        })

    result["abs_error"] = (result["actual"] - result["predicted"]).abs()

    result["RUL_range"] = pd.cut(
        result["actual"],
        bins=[-np.inf, 25, 50, 75, 100, 150, 200, np.inf]
        )

    print(result.groupby("RUL_range")["abs_error"].agg(["mean","std","count"])) 


display_error_by_RUL_range(y_val,y_val_pred)