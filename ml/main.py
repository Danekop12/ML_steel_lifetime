import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

Steel_Data = pd.read_excel("../data/NIMS_FATIGUE.xlsx")

# Tester konvertering til log10 av Nf verdiene i datasettet. Legger log10 verdiene som egen kolonne i datasettet
Steel_Data["Nf_log"] = np.log10(Steel_Data["Nf"])

print(Steel_Data.head())