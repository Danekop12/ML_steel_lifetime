# We run all models here. Run data_processing to get data and then try different models. Plot result and give evaluation metrics

from data_processing import process_data, split_data, scale_data
from data_modeling import (get_evaluation, plot_predictions,model_LinearRegression, model_SVR_linear, model_SVR_rbf, model_SVR_poly,model_RandomForest, model_XGBoost, model_PolynomialRegression, model_MLP_Regression)


# The model function to train: model_LinearRegression, model_SVR_linear, model_SVR_rbf, model_SVR_poly,
#                              model_RandomForest, model_XGBoost, model_PolynomialRegression, model_MLP_Regression
MODEL = model_XGBoost

# How much of the data is used for training, in percent
# TODO: write 80/20, 70/30, 90/10
SPLIT = 80

# "standard", "minmax" or "robust"
SCALER = "standard"

# Name used in prints and plot title, e.g. "XGBoost" from model_XGBoost
model_name = MODEL.__name__.removeprefix("model_")

# Process the raw data, split it in a training and a test set and scale it
Steel_Data = process_data()
X_train, X_test, y_train, y_test = split_data(Steel_Data, split=SPLIT)
X_train, X_test = scale_data(X_train, X_test, scaler_name=SCALER)

print("Model:", model_name)
print("Split (train/test): %d/%d" % (SPLIT, 100 - SPLIT))
print("Scaler:", SCALER)
print("Training rows:", len(X_train), "Test rows:", len(X_test))

# Train the chosen model and predict the test set
y_pred = MODEL(X_train, y_train, X_test)

# Calculate and print the evaluation metrics for the model on the test set, and plot the predictions
get_evaluation(model_name, y_test, y_pred)
plot_predictions(model_name, y_test, y_pred)