import numpy as np 
import pandas as pd 

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
)

#add column which contain max cycle of that unit
train_data['max_cycle'] = train_data.groupby('unit')['cycle'].transform('max')

#add column containing RUL for that unit
train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']

features = [
    col for col in train_data.columns if col not in ["unit","RUL","max_cycle"]
]

#clarify features for training and ground truth
x = train_data[features]
y = train_data["RUL"]


