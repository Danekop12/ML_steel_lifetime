# We run all models here. Run data_processing to get data and then try different models. Plot result and give evaluation metrics

from data_processing import process_data, split_data, scale_data
from data_modeling import MODELS, get_evaluation, plot_predictions


# Name of the model to train: "Linear regression", "SVR linear", "SVR rbf", "SVR poly",
#                             "Random forest", "XGBoost", "Polynomial regression", "MLP"
MODEL = "MLP"

# How much of the data is used for training, in percent
# TODO: write 80/20, 70/30, 90/10
SPLIT = 80

# "standard", "minmax" or "robust"
SCALER = "standard"

##############################

# Process the raw data, split it in a training and a test set and scale it
Steel_Data = process_data()
X_train, X_test, y_train, y_test = split_data(Steel_Data, split=SPLIT)
X_train, X_test = scale_data(X_train, X_test, scaler_name=SCALER)

print("Model:", MODEL)
print("Split (train/test): %d/%d" % (SPLIT, 100 - SPLIT))
print("Scaler:", SCALER)
print("Training rows:", len(X_train), "Test rows:", len(X_test))

#Find the function of the chosen model, train it and predict the test set
train_model = MODELS[MODEL]
y_pred = train_model(X_train, y_train, X_test)

# Calculate and print the evaluation metrics for the model on the test set, and plot the predictions
get_evaluation(MODEL, y_test, y_pred)
plot_predictions(MODEL, y_test, y_pred)