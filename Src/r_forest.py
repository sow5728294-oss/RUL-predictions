import pandas as pd
from pathlib import Path

#set base file location
BASE_DIR = Path(__file__).resolve().parent.parent

#set specific processed data location
processed_dir = BASE_DIR / "Data" / "target data" / "processed"


#load the training data
train_data = pd.read_csv(
    processed_dir/"train_processed.csv"
)

#seperate into input and answer
x_train = train_data.drop(columns=["unit", "RUL"])
y_train = train_data["RUL"]


#START CREATING MODEL

from sklearn.ensemble import RandomForestRegressor

#create the model
model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs = -1
)

#train the model

model.fit(x_train,y_train)