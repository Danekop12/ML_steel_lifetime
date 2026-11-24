# We run all models here. Run data_processing to get data and then try different models. Plot result and give evaluation metrics
import os
import pandas as pd
from data_processing import process_data, split_data, scale_data
from data_modeling import (get_evaluation, plot_predictions,model_LinearRegression, model_SVR_linear, model_SVR_rbf, model_SVR_poly,model_RandomForest, model_XGBoost, model_PolynomialRegression, model_MLP_Regression, model_LightGBM)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

COMMON_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_STEEL_PATH = os.path.join(COMMON_FOLDER, "..", "data", "NIMS_FATIGUE_processed.csv")


# The model function to train: model_LinearRegression, model_SVR_linear, model_SVR_rbf, model_SVR_poly,
#                              model_RandomForest, model_XGBoost, model_PolynomialRegression, model_MLP_Regression
model = model_LightGBM
model_name = model.__name__ 

# How much of the data is used for training, in percent
# TODO: write 80/20, 70/30, 90/10
SPLIT = 85

# Scalers: StandardScaler(), MinMaxScaler(), RobustScaler()
#scaler = StandardScaler()

def load_processed_data():
    Steel_Data = pd.read_csv(DATA_STEEL_PATH)
    return Steel_Data

# Process the raw data, split it in a training and a test set and scale it
Steel_Data = load_processed_data()
X_train, X_test, y_train, y_test = split_data(Steel_Data, split=SPLIT)
X_train, X_test = scale_data(X_train, X_test)

print("Model:", model_name)
print("Split (train/test): %d/%d" % (SPLIT, 100 - SPLIT))
print("Training rows:", len(X_train), "Test rows:", len(X_test))

# Train the chosen model and predict the test set
y_pred = model(X_train, y_train, X_test)

# Calculate and print the evaluation metrics for the model on the test set, and plot the predictions
get_evaluation(model_name, y_test, y_pred)
plot_predictions(model_name, y_test, y_pred)