import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from xgboost import XGBRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
import joblib 

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
#only run when needed
def test_param():

    model_arch = XGBRegressor(
        random_state = 42,
        n_jobs = -1
    )

    #list to choose hyperparameters from
    param_list = {
        "n_estimators":[300,500,700,900,1200,1500],
        "learning_rate":[0.01,0.02,0.03,0.05,0.07,0.1],
        "max_depth":[3,4,5,6,7,8],
        "subsample":[0.7,0.8,0.9,1.0],
        "colsample_bytree":[0.6,0.7,0.8,0.9,1.0],
        "gamma": [0, 0.1, 0.3, 0.5, 1],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
        "reg_lambda": [0.5, 1, 2, 5, 10]
    }
    
    cv = GroupKFold(n_splits=5)

    #try 90 different combination of hyperparameter and train them to evaluate the best combination
    search_model = RandomizedSearchCV(
    estimator=model_arch,
    param_distributions=param_list,
    n_iter=90,
    scoring="neg_mean_absolute_error",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

    search_model.fit(
        x_train,
        y_train,
        groups = train_data["unit"]
        )
    #print the best hyperparameters combination with the MAE score
    print("Best parameters:")
    print(search_model.best_params_)

    print("\nBest CV MAE:")
    print(-search_model.best_score_)

#implementation of the best hyperparameters combination
model = XGBRegressor(
    n_estimators = 900, # no need further reduce, already no overfitting
    learning_rate = 0.01, #increase wont help
    max_depth =8, #increase wont help 
    random_state = 42,
    subsample = 0.7,
    colsample_bytree = 0.6,
    reg_lambda = 2,
    reg_alpha = 0.1,
    gamma = 0.5
    )

#train the model 
model.fit(x_train,y_train)

#get prediction for x_train and x_val (the output is logged)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

#compare prediction y with actual y and calculate error 
train_mae = mean_absolute_error(y_train,y_train_pred)
val_mae = mean_absolute_error(y_val,y_val_pred)

#calculate RMSE of train and validation set
train_rmse = np.sqrt(
    mean_squared_error(y_train,y_train_pred)
)
val_rmse = np.sqrt(
    mean_squared_error(y_val,y_val_pred)
)

#calculate r2 score for train and validation set
train_r2 = r2_score(y_train,y_train_pred)
val_r2 = r2_score(y_val,y_val_pred)


#print error score for MAE, RMSE and r2
print(f"FOR TRAIN DATA: \n MAE: {train_mae} \n RMSE: {train_rmse} \n R2: {train_r2}")
print(f"FOR VALIDATION DATA: \n MAE: {val_mae} \n RMSE: {val_rmse} \n R2: {val_r2}")


#display amount of error in each RUL range
#run when needed
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

#save the trained model parameters 
model_path = BASE_DIR / "Model" / "xgb_model.pkl"
joblib.dump(model, model_path)

