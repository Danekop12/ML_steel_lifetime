import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

DATA_PATH = "../data/NIMS_FATIGUE.xlsx"
PROCESSED_PATH = "../data/NIMS_FATIGUE_processed.csv"

##############################
# Processing options. Change one at a time, run main.py and note the result.

# Only rows with Nf between these two values are used
NF_LOWER = 1e5
NF_UPPER = 1e7

# Run-outs are tests that were stopped without failure. They are logged at the stop value
# (584 rows at exactly 2.12e7 and 126 rows at exactly 1e7), so their Nf is not a real lifetime.
REMOVE_RUNOUTS = False

# Which heat treatment route to use:
#   "all" - all rows
#   "N"   - normalizing only
#   "NQT" - normalizing + quenching + tempering
#   "QT"  - quenching + tempering
HEAT_TREATMENT = "all"

# What to do with empty temperature values (empty means the step was not performed):
#   "zero"      - add a present/not present column and fill with 0
#   "median"    - add a present/not present column and fill with the median temperature
#   "drop_rows" - remove all rows with missing values (as in the paper)
MISSING_TEMP = "zero"

# Remove rows that are exact copies of another row
REMOVE_DUPLICATES = True

# Remove the 40 rows of S40C tempered at 180 °C. Their hardness (585) is far from all other rows (130-390).
REMOVE_LOW_TEMPERING = False

# Remove outliers in Nf_log with the IQR method
REMOVE_OUTLIERS = False

# How the steel grade is turned into numbers:
#   "onehot" - one 0/1 column for each grade
#   "carbon" - one column with the carbon content (S25C = 0.25)
MATERIAL_ENCODING = "onehot"

# Drop the inclusion columns dA, dB and dC (done in the paper)
DROP_INCLUSIONS = False

# Drop the four stress columns that are highly correlated with Vickers hardness
DROP_CORRELATED = True

# Extra features made from the existing columns
ADD_LOG_STRESS = False     # log10 of σa (Basquin's law: log Nf is close to linear in log σa)
ADD_STRESS_RATIO = False   # σa divided by Vickers hardness

##############################

RUNOUT_VALUES = [1e7, 2.12e7]
TEMP_COLUMNS = ["Normalizing temp", "Quenching temp", "Tempering temp"]
INCLUSION_COLUMNS = ["dA", "dB", "dC"]
CORRELATED_COLUMNS = ["Upper yield stress", "0.2% proof stress", "Tensile strength", "True fracture stress"]
CARBON_CONTENT = {"S25C": 0.25, "S35C": 0.35, "S40C": 0.40, "S45C": 0.45, "S55C": 0.55}

SCALERS = {
    "standard": StandardScaler(),   # No fixed range, sensitive to extreme outliers
    "minmax": MinMaxScaler(),       # Range [0, 1], outliers squeeze the rest of the values
    "robust": RobustScaler(),       # Uses median and IQR, less sensitive to outliers
}

# Columns that are not features
NON_FEATURE_COLUMNS = ["Nf", "Nf_log"]


def load_raw_data():
    Steel_Data = pd.read_excel(DATA_PATH)
    Steel_Data.columns = Steel_Data.columns.str.strip()
    return Steel_Data


def filter_heat_treatment(Steel_Data):
    # The route is found from which temperatures are filled in.
    # The result is a column with the text "N", "NQT" or "QT" for every row.
    normalized = Steel_Data["Normalizing temp"].notna()
    quenched = Steel_Data["Quenching temp"].notna()
    route = normalized.map({True: "N", False: ""}) + quenched.map({True: "QT", False: ""})

    keep = route == HEAT_TREATMENT
    return Steel_Data[keep]


def add_present_columns(Steel_Data):
    # New columns with 1 if the heat treatment step was performed and 0 if not
    Steel_Data["normalizing_temp_present"] = Steel_Data["Normalizing temp"].notna().astype(int)
    Steel_Data["quenching_temp_present"] = Steel_Data["Quenching temp"].notna().astype(int)
    Steel_Data["tempering_temp_present"] = Steel_Data["Tempering temp"].notna().astype(int)
    return Steel_Data


def remove_outliers_iqr(Steel_Data):
    q1 = Steel_Data["Nf_log"].quantile(0.25)
    q3 = Steel_Data["Nf_log"].quantile(0.75)
    iqr = q3 - q1
    keep = Steel_Data["Nf_log"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return Steel_Data[keep]


def remove_constant_columns(Steel_Data):
    # Columns with only one value give no information (happens when one heat treatment route is chosen)
    number_of_values = Steel_Data.nunique()
    constant_columns = number_of_values[number_of_values == 1].index
    return Steel_Data.drop(columns=constant_columns)


def process_data():
    Steel_Data = load_raw_data()
    print("Rows in the raw data:", len(Steel_Data))

    # Log10 of Nf is the target, because Nf spans from 1e4 to 1e10
    Steel_Data["Nf_log"] = np.log10(Steel_Data["Nf"])

    ##############################
    # Choose the rows.
    # The pattern is the same every time: first make a True/False value for every row,
    # then Steel_Data[keep] returns only the rows where the value is True.
    keep = Steel_Data["Nf"].between(NF_LOWER, NF_UPPER)
    Steel_Data = Steel_Data[keep]
    print("Rows after the Nf range filter:", len(Steel_Data))

    if REMOVE_RUNOUTS:
        is_runout = Steel_Data["Nf"].isin(RUNOUT_VALUES)
        Steel_Data = Steel_Data[~is_runout]     # ~ means "not": keep the rows that are not run-outs
        print("Rows after removing run-outs:", len(Steel_Data))

    if HEAT_TREATMENT != "all":
        Steel_Data = filter_heat_treatment(Steel_Data)
        print("Rows after the heat treatment filter:", len(Steel_Data))

    if REMOVE_LOW_TEMPERING:
        is_low_tempering = Steel_Data["Tempering temp"] < 500
        Steel_Data = Steel_Data[~is_low_tempering]
        print("Rows after removing low tempering:", len(Steel_Data))

    if REMOVE_OUTLIERS:
        Steel_Data = remove_outliers_iqr(Steel_Data)
        print("Rows after removing outliers:", len(Steel_Data))

    if REMOVE_DUPLICATES:
        Steel_Data = Steel_Data.drop_duplicates()
        print("Rows after removing duplicates:", len(Steel_Data))

    ##############################
    # Choose the columns
    # "Heat" and "DS No. PDF" are ids and not physical properties
    Steel_Data = Steel_Data.drop(columns=["Heat", "DS No. PDF"])

    if DROP_CORRELATED:
        Steel_Data = Steel_Data.drop(columns=CORRELATED_COLUMNS)

    if DROP_INCLUSIONS:
        Steel_Data = Steel_Data.drop(columns=INCLUSION_COLUMNS)

    ##############################
    # Missing temperatures (empty means the step was not performed)
    if MISSING_TEMP == "zero":
        Steel_Data = add_present_columns(Steel_Data)
        Steel_Data[TEMP_COLUMNS] = Steel_Data[TEMP_COLUMNS].fillna(0)

    if MISSING_TEMP == "median":
        Steel_Data = add_present_columns(Steel_Data)
        Steel_Data[TEMP_COLUMNS] = Steel_Data[TEMP_COLUMNS].fillna(Steel_Data[TEMP_COLUMNS].median())
        # A column that is completely empty has no median, so it is filled with 0
        Steel_Data[TEMP_COLUMNS] = Steel_Data[TEMP_COLUMNS].fillna(0)

    # Remove the rows that still have missing values. With "drop_rows" this removes every row with
    # an empty temperature. If the stress columns are kept, it removes the rows where they are empty.
    Steel_Data = Steel_Data.dropna()

    ##############################
    # Turn the text columns into numbers
    if MATERIAL_ENCODING == "onehot":
        Steel_Data = pd.get_dummies(Steel_Data, columns=["Testing Material"], dtype=int)

    if MATERIAL_ENCODING == "carbon":
        Steel_Data["Carbon content"] = Steel_Data["Testing Material"].map(CARBON_CONTENT)
        Steel_Data = Steel_Data.drop(columns=["Testing Material"])

    Steel_Data = pd.get_dummies(Steel_Data, columns=["Type of test"], dtype=int)

    ##############################
    # Extra features
    if ADD_LOG_STRESS:
        Steel_Data["σa_log"] = np.log10(Steel_Data["σa"])

    if ADD_STRESS_RATIO:
        Steel_Data["σa / Vickers hardness"] = Steel_Data["σa"] / Steel_Data["Vickers hardness"]

    Steel_Data = remove_constant_columns(Steel_Data)
    Steel_Data = Steel_Data.reset_index(drop=True)
    print("Processed data:", len(Steel_Data), "rows and", Steel_Data.shape[1] - len(NON_FEATURE_COLUMNS), "features")

    # Save the processed data to be able to look at it
    Steel_Data.to_csv(PROCESSED_PATH, index=False)
    return Steel_Data


def split_data(Steel_Data, split=80):
    X = Steel_Data.drop(columns=NON_FEATURE_COLUMNS)
    y = Steel_Data["Nf_log"]

    # Split the data into training and testing sets, with reproducible splits using random_state = 42
    test_size = (100 - split) / 100
    return train_test_split(X, y, test_size=test_size, random_state=42)


def scale_data(X_train, X_test, scaler_name="standard"):
    scaler = SCALERS[scaler_name]

    # Scale all columns except the 0/1 columns
    number_of_values = X_train.nunique()
    columns = number_of_values[number_of_values > 2].index

    # The scaler is fitted on the training rows only, so it learns nothing from the test rows
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    Steel_Data = process_data()
    print(Steel_Data.head())
    print(Steel_Data.describe())