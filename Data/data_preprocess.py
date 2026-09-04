import numpy as np 
import pandas as pd 

train_data = pd.read_csv(
    "target data/train_FD002.txt",
    sep = r"\s+",
    header = None
)

print(train_data)

