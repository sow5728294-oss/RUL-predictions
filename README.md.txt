MACHINE LEARNING MODEL FOR PREDICTING REMAINING USEFUL LIFE (RUL) OF TURBOFAN ENGINE


Overview
This project develops a machine learning model to predict the Remaining Useful Life (RUL) of a turbofan engine.

The model receives data from a turbofan engine unit, including its cycle, 3 operational settings, and 21 sensor measurements, and predicts how many cycles of engine life are left until failure.

The model is aimed to be used by aircraft maintenance ground crew or relevant positions in the aviation field to get a brief idea of the engine's current status.

It provides a time-efficient way to briefly determine the turbofan engine status before performing a more detailed analysis of the large amount of engine data.


Dataset
Source: NASA C-MAPSS-1 Turbofan Engine Degradation Dataset
Number of samples: 260 train trajectories with 53,759 rows of data
Number of features:
26 features in the raw data
214 features in the processed data
Target variable: RUL
Train/Test split:
Train and Test datasets contain the same amount of trajectories
20% of the trajectories in the training dataset are used as the validation set


Methodology
The overall workflow of the project is:

Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Prediction


Models
Two machine learning models were developed and evaluated:

Random Forest
XGBoost
These models were chosen as they are suitable for handling structured/tabular data and can capture complex relationships between the engine sensor measurements and its Remaining Useful Life.

The XGBoost model was selected as the final model because it achieved better overall performance than the Random Forest model on the validation set.

As a result, the test set was only evaluated using the XGBoost model.

Results
Validation Set
Metric	Score
MAE	26.13
RMSE	34.63
R² Score	0.73

Test Set
Metric	Score
MAE	33.62
RMSE	42.74
R² Score	0.55

The test set was only run using the XGBoost model, as its overall performance was better than the Random Forest model on the validation set.

How to Run
1. Clone the Repository
git clone ...

2. Install Dependencies
pip install -r requirements.txt

3. Save the Dataset
Save the dataset .txt files in:

Data/target data/raw

Edit the Train_path, test_path, and RUL_path variables in data_preprocess.py to match the location of your dataset.

4. Run the Code to Train the Model
Run the following commands:

python data_preprocess.py
python xgb.py
python test_model.py

Limitations
There is a large gap between the test set accuracy and validation set accuracy.
The model accuracy is currently low and is not suitable for actual application.
The accuracy decreases rapidly for high RUL predictions (RUL > 200).
Future Improvements
Better model selection.
A larger dataset with more data for high RUL prediction.
Better feature engineering to capture the sudden degradation of the engine as model input.


Project Structure
Data/
└── All downloaded data from NASA data source
    └── target data
        ├── raw
        └── processed

Model/
└── Parameters and hyperparameters of the trained XGBoost model

Src/
├── data_preprocess.py
├── r_forest.py
├── test_model.py
└── xgb.py

Data
The Data folder contains all downloaded data from the NASA data source.

The target data folder contains the FD002 dataset, which is used to train this model. The target data is separated into:

raw — Raw dataset downloaded from the NASA data source.
processed — Processed dataset generated after data preprocessing and feature engineering.
Model
The Model folder contains the parameters and hyperparameters of the trained XGBoost model.

Src
The Src folder contains the source code for several stages of the model development:

data_preprocess.py — Processes the raw data and saves the processed data into target data/processed.
r_forest.py — Code used for the development of the Random Forest machine learning model, which is not used as the final model.
test_model.py — Tests the XGBoost model and calculates the MAE, RMSE, and R² Score.
xgb.py — Code used for training and development of the XGBoost machine learning model.



