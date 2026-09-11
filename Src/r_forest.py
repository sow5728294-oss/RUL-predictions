import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib

#set base file location
BASE_DIR = Path(__file__).resolve().parent.parent

#set specific processed data location
processed_dir = BASE_DIR / "Data" / "target data" / "processed"


#load the training data
train_data = pd.read_csv(
    processed_dir/"train_processed.csv"
)

#load validation data
val_data = pd.read_csv(
    processed_dir/"val_processed.csv"
)



#seperate into input and answer
x_train = train_data.drop(columns=["unit", "RUL"])
y_train = train_data["RUL"]

#seperate into input and answer for val set
x_val = val_data.drop(columns=["unit", "RUL"])
y_val = val_data["RUL"]

#MODEL FOR RANDOM HYPERPARAMETER TO FIND BEST PARAMATERS COMBINATION
#only run when needed
def test_param():
        #MODEL FOR RANDOM HYPERPARAMETER TO FIND BEST PARAMATERS COMBINATION
        #create the model
        model_arch = RandomForestRegressor(
            random_state = 42,
            n_jobs = -1
        )

        #craete list of different hyperparameters to test from
        param_list = {
            "n_estimators":[100,200,300,400,500,600,700],
            "max_depth":[5,10,15,20,25,30,40,60],
            "min_samples_split":[2,5,10,15,20],
            "min_samples_leaf":[1,2,4,6],
            "max_features":[0.5,0.7,1.0,"sqrt","Log2"]
        }

        #create model with randomized hyperparameter search
        diff_comb_model = RandomizedSearchCV(
            estimator = model_arch,
            param_distributions = param_list,
            n_iter = 100,
            cv = 5,
            scoring = "neg_mean_absolute_error",
            random_state = 42,
            n_jobs = -1
        )

        #train the model
        diff_comb_model.fit(x_train,y_train)

        print(diff_comb_model.best_params_)
        

#IMPLEMENTATION OF THE BEST HYPERPARAMETERS
model = RandomForestRegressor(
    n_estimators = 500,
    max_depth = 30,
    min_samples_split = 15,
    min_samples_leaf = 8,
    max_features = "log2",
    random_state = 42 
)

#train model
model.fit(x_train,y_train)


#make prediction
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)


#calculate the MAE
val_mae = mean_absolute_error(y_val,y_val_pred)
train_mae = mean_absolute_error(y_train,y_train_pred)

#calculate RMSE
val_rmse = np.sqrt(
    mean_squared_error(y_val,y_val_pred)
)
train_rmse = np.sqrt(
    mean_squared_error(y_train,y_train_pred)
)

#calculate r2 square
val_r2 = r2_score(y_val,y_val_pred)
train_r2 = r2_score(y_train,y_train_pred)

#print result
print(f" FOR TRAINING SET \n mae: {train_mae}\nrmse: {train_rmse}\nr2: {train_r2}")
print(f" FOR VAL SET \n mae: {val_mae}\nrmse: {val_rmse}\nr2: {val_r2}")

#save the trained model parameters 
model_path = BASE_DIR / "Model" / "r_forest_model.pkl"
joblib.dump(model, model_path)