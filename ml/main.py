from data_processing import process_data

Steel_Data, Steel_Data3, Steel_Data_Scaled, groups = process_data()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def model():
    # Define the features and target variable of the dataset. Here, we are using all columns except 'Nf' and 'Nf_log' as features, and 'Nf_log' as the target variable.
    X = Steel_Data_Scaled.drop(columns=['Nf', 'Nf_log'])
    y = Steel_Data_Scaled['Nf_log']  # Using log-transformed Nf as the target variable

    # Split the data into training and testing sets, with reproducible splits using random_state = 42 (the meaning of life, the universe, and everything)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and applya linear regression model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    # Print the coefficients and intercept of the model
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the evaluation metrics for the model pn the test set
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Mean absolute error: %.2f" % mae)
    print("Mean squared error: %.2f" % mse)
    print("Residual sum of squares (MSE): %.2f" % mse ** 2)
    print("R2-score: %.2f" % r2)

    # Plot the actual values against the predicted values on a log-log scale to visualize the model's performance
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
   
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    #lt.axis('equal')
    plt.show()

model()