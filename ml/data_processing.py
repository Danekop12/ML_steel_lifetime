import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

COMMON_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_STEEL_PATH = os.path.join(COMMON_FOLDER, "..", "data", "NIMS_FATIGUE.xlsx")
DATA_PROCESSED_PATH = os.path.join(COMMON_FOLDER, "..", "data", "NIMS_FATIGUE_processed.csv")

# Range of Nf to keep. Filtered after applying log10 function, so in range of 5 to 7. Same as 10^5 and 10^7 before
NF_MIN = 4
NF_MAX = 6.99

#List of columns to remove from dataset.
# All columns: "dA", "dB", "dC", "Upper yield stress", "0.2% proof stress", "Tensile strength", "True fracture stress", "Normalizing temp", "Quenching temp", "Tempering temp", "σa / Vickers hardness", "σa", "Vickers hardness", "Testing Material", "Type of test", "Nf", "Nf_log"
REMOVE_COLUMNS = ["dA", "dB", "dC"]
# Colums to remove, have high correlation with Vickers hardness
CORRELATED_COLUMNS = ["Upper yield stress", "0.2% proof stress", "Tensile strength", "True fracture stress"]

# Choose which heat to model: "all" - all rows,
#  N - normalizing
#  NQT - normalizing + quenching + tempering
#  QT - quenching + tempering
HEAT_TREATMENT = "all"


# Drop the four stress columns that are highly correlated with Vickers hardness
DROP_CORRELATED = True

TEMP_COLUMNS = ["Normalizing temp", "Quenching temp", "Tempering temp"]

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", None)

# Chose scaler: StandardScaler(), MinMaxScaler(), RobustScaler()
scaler = StandardScaler()


def load_unprocessed_data():
    Steel_Data = pd.read_excel(DATA_STEEL_PATH)
    # Fix column "0.2% proof stress" spacing
    Steel_Data.columns = Steel_Data.columns.str.strip()
    Steel_Data = Steel_Data.drop(columns=["Heat", "DS No. PDF"])
    return Steel_Data

def split_data(Steel_Data, split):
    # Convert to persentage for relative value. Int is amount of data rows
    split_percentage = split / 100
    X = Steel_Data.drop(columns=["Nf", "Nf_log"])
    Y = Steel_Data["Nf_log"]

    test_size = 1 - split_percentage
    print(f"Split (train/test): {split}/{100 - split}")
    return train_test_split(X, Y, test_size=test_size, random_state=42)


# Convert Nf to log10(Nf) and add it as a new column
def log10_data(Steel_Data):
    Steel_Data["Nf_log"] = np.log10(Steel_Data["Nf"])
    return Steel_Data

# Filters Dataset by checking if temperatures are present or not. 
# N, NQT, QT. Returns the filtered dataset, is_temp dropped
def filter_heat_treatment(Steel_Data):
    normalized = (Steel_Data["n_temp_present"] == 1)
    not_normalized = (Steel_Data["n_temp_present"] == 0)
    quenched = (Steel_Data["q_temp_present"] == 1)
    not_quenched = (Steel_Data["q_temp_present"] == 0)
    if HEAT_TREATMENT == "N":
        Steel_Data = Steel_Data[normalized & not_quenched]
        Steel_Data = Steel_Data.drop(columns=["Tempering temp", "Quenching temp"])
    elif HEAT_TREATMENT == "NQT":
        Steel_Data = Steel_Data[normalized & quenched]
    elif HEAT_TREATMENT == "QT":
        Steel_Data = Steel_Data[not_normalized & quenched]
        Steel_Data = Steel_Data.drop(columns=["Normalizing temp"])
    # Drop is_present columns, they are redundant
    Steel_Data= Steel_Data.drop(columns=["n_temp_present", "q_temp_present", "t_temp_present"])
    return Steel_Data

# Extra columns, shows if the temperature columns are present or not: 1 if present, 0 if not.
def add_present_columns(Steel_Data):
    Steel_Data["n_temp_present"] = Steel_Data["Normalizing temp"].notna().astype(int)
    Steel_Data["q_temp_present"] = Steel_Data["Quenching temp"].notna().astype(int)
    Steel_Data["t_temp_present"] = Steel_Data["Tempering temp"].notna().astype(int)
    return Steel_Data

 # Convert text columns to numbers
def encode_text_columns(Steel_Data):
    Steel_Data = pd.get_dummies(Steel_Data, columns=["Type of test"], drop_first=True, dtype="int")
    Steel_Data = pd.get_dummies(Steel_Data, columns=["Testing Material"], drop_first=True, dtype="int", prefix="Material")
    return Steel_Data

def process_data():
    Steel_Data = load_unprocessed_data()
    Steel_Data = Steel_Data.drop(columns=REMOVE_COLUMNS)
    # Convert Nf to log10(Nf)
    Steel_Data = log10_data(Steel_Data)
    #Filters Nf log by val described earlier
    Steel_Data = Steel_Data[(Steel_Data["Nf_log"] >= NF_MIN) & (Steel_Data["Nf_log"] <= NF_MAX)]
    print(Steel_Data["Nf_log"].describe())  # Show the range of Nf_log after filtering

    # To try with all colums too
    if DROP_CORRELATED == True:
        Steel_Data = Steel_Data.drop(columns=CORRELATED_COLUMNS)
    # Set 0 for missing temperatures
    Steel_Data = add_present_columns(Steel_Data)
    Steel_Data[TEMP_COLUMNS] = Steel_Data[TEMP_COLUMNS].fillna(0)
    
    if HEAT_TREATMENT != "all":
        Steel_Data = filter_heat_treatment(Steel_Data)
        print("Rows after temperature filter:", len(Steel_Data))
    
    # Remove the rows that still have missing values. With "drop_rows" this removes every row with an empty temperature. If the stress columns are kept, it removes the rows where they are empty.
    #Steel_Data = Steel_Data.dropna()
    
    # Text converted to numbers, one-hot encoding
    Steel_Data = encode_text_columns(Steel_Data)   
    
    
    
    
    # -2 for Nf and Nf_log
    amount_of_features = Steel_Data.shape[1] - 2
    print("Data rows after processing: ", len(Steel_Data))
    print("Amount of features: ", amount_of_features)
    print(Steel_Data.head())    
    print(Steel_Data.describe())
    # Save processed data
    Steel_Data.to_csv(DATA_PROCESSED_PATH, index=False)

def scale_data(X_train, X_test):
    number_of_values = X_train.nunique()
    columns = number_of_values[number_of_values > 2].index

    X_train_fit = X_train.copy()
    X_test_fit = X_test.copy()
    X_train_fit[columns] = scaler.fit_transform(X_train[columns])
    X_test_fit[columns] = scaler.transform(X_test[columns])
    return X_train_fit, X_test_fit


if __name__ == "__main__":
    process_data()