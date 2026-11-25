# We run all models here. Run data_processing to get data and then try different models. Plot result and give evaluation metrics
import os
import pandas as pd
from data_processing import process_data, split_data, scale_data
from data_modeling import (get_evaluation, plot_predictions, model_LinearRegression, model_SVR_linear, model_SVR_rbf, model_SVR_poly, model_XGBoost, model_PolynomialRegression, model_MLP, model_LightGBM)

COMMON_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_STEEL_PATH = os.path.join(COMMON_FOLDER, "..", "data", "NIMS_FATIGUE_processed.csv")

# The model function to use in training and testing
# Use these keywords below to select a model: model_LinearRegression, model_SVR_linear, model_SVR_rbf, model_SVR_poly,
#                                             model_XGBoost, model_PolynomialRegression, model_MLP, model_LightGBM
model = model_XGBoost
model_name = model.__name__ 

# How much of the data is used for training, in percent
# TODO: write 85/15, 80/20, 70/30, 90/10
SPLIT = 85

# Function to load the processed data from a CSV file. This function reads the CSV file containing the processed steel data and returns it as a pandas DataFrame.
def load_processed_data():
    Steel_Data = pd.read_csv(DATA_STEEL_PATH)
    return Steel_Data

# Load the processed data and process, split in train and test sets, and scale the features of the sets.
Steel_Data = load_processed_data()
X_train, X_test, y_train, y_test = split_data(Steel_Data, split=SPLIT)
X_train, X_test = scale_data(X_train, X_test)

# Print output information about the model being used and the number of training and test rows.
print("Model:", model_name)
print("Training rows:", len(X_train), "Test rows:", len(X_test))

# Use the selected model to train on the training set and make predictions on the test set.
y_pred = model(X_train, y_train, X_test)

# Use the predicted values from the model and the actual values of the dataset to calculate evaluation metrics and plot prediction curve
get_evaluation(model_name, y_test, y_pred)
plot_predictions(model_name, y_test, y_pred)