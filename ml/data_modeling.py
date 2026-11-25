import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm

# Random seed for reproducibility, used for the models that have random elements in their training process, such as MLPRegressor, XGBoost, and LightGBM.
# This ensures that the results are consistent across different runs of the code.
RANDOM_SEED = 42

# Function to give the MAE, MSE, RMSE and R2-score for the model on the test set.
def get_evaluation(name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Evaluation metrics for", name, " :")
    print(f"Mean absolute error: {mae:.3f}")
    print(f"Mean squared error: {mse:.3f}")
    print(f"Root mean squared error: {rmse:.3f}")
    print(f"R2-score: {r2:.3f}")

# Visualize the predictied Nf values (from the model) vs the actual Nf values (from the dataset). 
def plot_predictions(name, y_test, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, c='crimson', alpha=0.5)
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual values:')
    plt.ylabel('Predicted values:')
    plt.title(name)
    plt.show()

# Function for the Linear Regression model in the setup
# Based on the LinearRegression from sklearn
# This is our benchmark model, used to compare other models as a basic step
def model_LinearRegression(X_train, y_train, X_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

#
# Will not use this model in project report or presentation.
#
def model_SVR_linear(X_train, y_train, X_test):
    model = SVR(kernel="linear")
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Function for the SVR model with RBF kernel in the setup
# Based on the SVR from sklearn
def model_SVR_rbf(X_train, y_train, X_test):
    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)
    return model.predict(X_test)
    # Got better results than poly 

#
# Will not use this model in project report or presentation.
#
def model_SVR_poly(X_train, y_train, X_test):
    model = SVR(kernel="poly")
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Function for the XGBoost model in the setup
# Based on the XGBRegressor from xgboost
def model_XGBoost(X_train, y_train, X_test):
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    #xgb.plot_importance(model, title ="XGBoost features importance")
    return model.predict(X_test)

# Function for the LightGBM model in the setup
# Based on the LGBMRegressor from lightgbm
def model_LightGBM(X_train, y_train, X_test):
    model = lightgbm.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=RANDOM_SEED, num_leaves=15)
    model.fit(X_train, y_train)
    #lightgbm.plot_importance(model, title="LightGBM features importance")
    return model.predict(X_test)

#
# Will not use this model in project report or presentation.
#
def model_PolynomialRegression(X_train, y_train, X_test, degree=3, alpha=10):
    expand = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = expand.fit_transform(X_train)
    X_test_poly = expand.transform(X_test)
    model = Ridge(alpha=alpha) # Regression with a penalty for large weights because there are more columns than rows
    model.fit(X_train_poly, y_train)
    print("Polynomial degree: ", degree)
    print("Alpha value: ", alpha)
    return model.predict(X_test_poly)

# Function for the MPL model in the setup
# Based on the MLPRegressor from sklearn
def model_MLP(X_train, y_train, X_test, hidden_layer_sizes=(256, 256, 256, 256), alpha=0.0001):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam", learning_rate_init=0.001,
                         alpha=alpha, max_iter=1500, batch_size=64, random_state=RANDOM_SEED, early_stopping=True,
                         validation_fraction=0.2, n_iter_no_change=50)
    model.fit(X_train, y_train)
    print("Hidden layer: ", hidden_layer_sizes)
    print("Alpha value: ", alpha)
    return model.predict(X_test)