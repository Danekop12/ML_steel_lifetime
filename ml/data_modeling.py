import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Every model function trains on the training set and returns the predictions for the test set.

RANDOM_SEED = 42


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


def plot_predictions(name, y_test, y_pred):
    # Plot the actual values against the predicted values to visualize the model's performance
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, c='crimson', alpha=0.5)
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual values (Nf_log)')
    plt.ylabel('Predicted values (Nf_log)')
    plt.title(name)
    plt.show()


def model_LinearRegression(X_train, y_train, X_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def model_SVR_linear(X_train, y_train, X_test):
    model = SVR(kernel="linear")
    model.fit(X_train, y_train)
    return model.predict(X_test)


def model_SVR_rbf(X_train, y_train, X_test):
    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)
    return model.predict(X_test)


def model_SVR_poly(X_train, y_train, X_test):
    model = SVR(kernel="poly")
    model.fit(X_train, y_train)
    return model.predict(X_test)


def model_RandomForest(X_train, y_train, X_test, n_estimators=300, min_samples_leaf=2):
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def model_XGBoost(X_train, y_train, X_test, n_estimators=400, max_depth=4, learning_rate=0.05):
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=RANDOM_SEED)
    model.fit(np.asarray(X_train), y_train)
    return model.predict(np.asarray(X_test))


def model_PolynomialRegression(X_train, y_train, X_test, degree=3, l2=0.001):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    expand = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = expand.fit_transform(X_train).astype("float32")
    X_test_poly = expand.transform(X_test).astype("float32")
    y_train_array = np.asarray(y_train, dtype="float32")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_poly.shape[1],)),
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    history = model.fit(X_train_poly, y_train_array, epochs=1000, batch_size=128,
                        validation_split=0.2, verbose=0)
    print("Epochs run:", len(history.history["loss"]), "of", 1000)

    return model.predict(X_test_poly, verbose=0).ravel()


def model_MLP_Regression(X_train, y_train, X_test, hidden_layers=(256, 256, 256, 256), dropout=0.3, learning_rate=0.001, epochs=500, batch_size=64, patience=50):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    X_train_array = np.asarray(X_train, dtype="float32")
    X_test_array = np.asarray(X_test, dtype="float32")
    y_train_array = np.asarray(y_train, dtype="float32")

    layers = [tf.keras.layers.Input(shape=(X_train_array.shape[1],))]
    for units in hidden_layers:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(X_train_array, y_train_array, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[early_stop], verbose=0)
    print("Epochs run:", len(history.history["loss"]), "of", epochs)

    return model.predict(X_test_array, verbose=0).ravel()


# The name of every model and the function that trains it. main.py picks one of them by name.
MODELS = {
    "Linear regression": model_LinearRegression,
    "SVR linear": model_SVR_linear,
    "SVR rbf": model_SVR_rbf,
    "SVR poly": model_SVR_poly,
    "Random forest": model_RandomForest,
    "XGBoost": model_XGBoost,
    "Polynomial regression": model_PolynomialRegression,
    "MLP": model_MLP_Regression,
}