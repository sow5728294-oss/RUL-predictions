import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data ():
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
        "target data/train_FD002.txt",
        sep = r"\s+",
        header = None,
        names = columns
    )

    #get test data
    test_data = pd.read_csv(
        "target data/test_FD002.txt",
        sep = r"\s+",
        header = None,
        names = columns
    )

    #get ground thruth(RUL) for test data
    RUL = pd.read_csv(
        "target data/RUL_FD002.txt",
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

    return (
            x_train_scaled.shape,
            x_val_scaled.shape,
            x_test_scaled.shape,
            y_train.shape,
            y_val.shape,
            RUL.shape,
            scaler
        )

print(load_data())