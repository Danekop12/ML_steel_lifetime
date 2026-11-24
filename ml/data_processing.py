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

# Columns that are not features
NON_FEATURE_COLUMNS = ["Nf", "Nf_log"]



def load_unprocessed_data():
    Steel_Data = pd.read_excel(DATA_STEEL_PATH)
    # Fix column "0.2% proof stress" spacing
    Steel_Data.columns = Steel_Data.columns.str.strip()
    Steel_Data = Steel_Data.drop(columns=["Heat", "DS No. PDF"])
    return Steel_Data

# Convert Nf to log10(Nf) and add it as a new column
def log10_data(Steel_Data):
    Steel_Data["Nf_log"] = np.log10(Steel_Data["Nf"])
    return Steel_Data

# Filters Dataset by checking if temperatures are present or not. 
# N, NQT, QT. Returns the filtered dataset, rest of rows are dropped.
def filter_heat_treatment(Steel_Data):
    normalized = Steel_Data["Normalizing temp"].notna()
    quenched = Steel_Data["Quenching temp"].notna()
    if HEAT_TREATMENT == "N":
        route = normalized.map({True: "N", False: ""})
    elif HEAT_TREATMENT == "NQT":
        route = normalized.map({True: "N", False: ""}) + quenched.map({True: "QT", False: ""})
    elif HEAT_TREATMENT == "QT":
        route = normalized.map({True: "N", False: ""}) + quenched.map({True: "QT", False: ""})
    # Choose rows to keep in the dataset
    keep = route.map({HEAT_TREATMENT: True, "": False})
    return Steel_Data[keep]



# Extra columns, shows if the temperature columns are present or not: 1 if present, 0 if not.
def add_present_columns(Steel_Data):
    Steel_Data["n_temp_present"] = Steel_Data["Normalizing temp"].notna().astype(int)
    Steel_Data["q_temp_present"] = Steel_Data["Quenching temp"].notna().astype(int)
    Steel_Data["t_temp_present"] = Steel_Data["Tempering temp"].notna().astype(int)
    return Steel_Data


def process_data():
    Steel_Data = load_unprocessed_data()
    Steel_Data = Steel_Data.drop(columns=REMOVE_COLUMNS)
    # Convert Nf to log10(Nf)
    Steel_Data = log10_data(Steel_Data)
    #Filters Nf log by val described earlier
    Steel_Data = Steel_Data[(Steel_Data["Nf_log"] >= NF_MIN) & (Steel_Data["Nf_log"] <= NF_MAX)]
    print(Steel_Data["Nf_log"].describe())  # Show the range of Nf_log after filtering
    if HEAT_TREATMENT != "all":
        Steel_Data = filter_heat_treatment(Steel_Data)
        print("Rows after temperature filter:", len(Steel_Data))
    
    # To try with all colums too
    if DROP_CORRELATED == True:
        Steel_Data = Steel_Data.drop(columns=CORRELATED_COLUMNS)
    # Set 0 for missing temperatures.
    Steel_Data = add_present_columns(Steel_Data)
    Steel_Data[TEMP_COLUMNS] = Steel_Data[TEMP_COLUMNS].fillna(0)
    # Remove the rows that still have missing values. With "drop_rows" this removes every row with an empty temperature. If the stress columns are kept, it removes the rows where they are empty.
    #Steel_Data = Steel_Data.dropna()
    
    # Turn the text columns into numbers
    Steel_Data = pd.get_dummies(Steel_Data, columns=["Testing Material"], dtype=int)
    Steel_Data = pd.get_dummies(Steel_Data, columns=["Type of test"], dtype=int)

    amount_of_features = Steel_Data.shape[1] - len(NON_FEATURE_COLUMNS)
    print("Processed data:", len(Steel_Data), "rows and", amount_of_features, "features")
    print(Steel_Data.head())
    print(Steel_Data.describe())
    # Save processed data
    Steel_Data.to_csv(DATA_PROCESSED_PATH, index=False)


def split_data(Steel_Data, split=80):
    X = Steel_Data.drop(columns=NON_FEATURE_COLUMNS)
    y = Steel_Data["Nf_log"]

    # Split the data into training and testing sets, with reproducible splits using random_state = 42
    test_size = (100 - split) / 100
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Scale only X train and test, not Y 
def scale_data(X_train, X_test):
    # Scale all columns except the 0/1 columns
    number_of_values = X_train.nunique()
    columns = number_of_values[number_of_values > 2].index
    # Scale training and testing data
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test

if __name__ == "__main__":
    process_data()