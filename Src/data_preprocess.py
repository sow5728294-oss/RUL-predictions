import numpy as np 
import pandas as pd 
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


processed_dir = BASE_DIR / "Data" / "target data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

def load_data ():

    train_path = BASE_DIR / "Data" / "target data" / "raw" / "train_FD002.txt"
    test_path = BASE_DIR / "Data" / "target data" / "raw" / "test_FD002.txt"
    RUL_path = BASE_DIR / "Data" / "target data" / "raw" / "RUL_FD002.txt"

    #assign all columns name
    columns = [
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_10",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_16",
        "sensor_17",
        "sensor_18",
        "sensor_19",
        "sensor_20",
        "sensor_21"   
    ]

    #get training data
    train_data = pd.read_csv(
       train_path,
        sep = r"\s+",
        header = None,
        names = columns
    )

    #get test data
    test_data = pd.read_csv(
        test_path,
        sep = r"\s+",
        header = None,
        names = columns
    )

    #get ground thruth(RUL) for test data
    RUL = pd.read_csv(
        RUL_path,
        sep = r"\s+",
        header = None,
    ).squeeze()

    #add column which contain max cycle of that unit
    train_data['max_cycle'] = train_data.groupby('unit')['cycle'].transform('max')

    #add column containing RUL for that unit
    train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']

    features = [
        col for col in train_data.columns if col not in ["unit","RUL","max_cycle"]
    ]


    #TRAIN VALIDATION SPLIT

    #Create array containing only engine units
    units = train_data["unit"].unique()

    #split unit array into train val ratio
    train_units, val_units = train_test_split(
        units,
        test_size=0.2,
        random_state=42
    )

    #split the train data into train and val with no mixing engine units
    train_byunit= train_data[train_data['unit'].isin(train_units)]
    val_byunit = train_data[train_data['unit'].isin(val_units)]

    #final x train and x val with only wanted columns
    x_train = train_byunit[features]
    x_val = val_byunit[features]
    y_train = train_byunit['RUL']
    y_val = val_byunit['RUL']

    #TEST DATA
    
    x_test = test_data[features]

    #SCALING DATA
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    x_val_scaled = scaler.transform(x_val)

    x_test_scaled = scaler.transform(x_test)

    # -----------------------------
    # Create processed DataFrames
    # -----------------------------

    processed_train = train_byunit[
        ["unit", "cycle"]
    ].copy()

    processed_train[features] = x_train_scaled

    processed_train["RUL"] = y_train.values


    processed_val = val_byunit[
        ["unit", "cycle"]
    ].copy()

    processed_val[features] = x_val_scaled

    processed_val["RUL"] = y_val.values


    processed_test = test_data[
        ["unit", "cycle"]
    ].copy()

    processed_test[features] = x_test_scaled

    # -----------------------------
    # Save CSV files
    # -----------------------------

    processed_train.to_csv(
        processed_dir / "train_processed.csv",
        index=False
    )

    processed_val.to_csv(
        processed_dir / "val_processed.csv",
        index=False
    )

    processed_test.to_csv(
        processed_dir / "test_processed.csv",
        index=False
    )

    # Save scaler for future predictions
    joblib.dump(
        scaler,
        processed_dir / "scaler.pkl"
    )

    return (
        processed_train,
        processed_val,
        processed_test,
        RUL,
        scaler
    )


if __name__ == "__main__":
    load_data()
   