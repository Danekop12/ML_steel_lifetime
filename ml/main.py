import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

Steel_Data = pd.read_excel("../data/NIMS_FATIGUE.xlsx")

print(Steel_Data.head())