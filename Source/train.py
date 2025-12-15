# Importer
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Funksjon som splitter i trening og test data
def split_data(X, y, test_size = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=1)
    return X_train, X_test, y_train, y_test

# -------------------------------------
# Funksjoner som trener modellene

# En funksjon som trener modellen
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=1, max_depth=8,n_estimators=500, min_samples_split=5)
    model.fit(X_train, y_train)
    return model

# Lineær regresjons modell
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ---------------------------------------

# En funksjon som utfører prediksjoner
def make_predictions(model, X_test):
    y_test_pred = model.predict(X_test)
    return y_test_pred

# ----------------------------------------
# Evaluer modellen
def evaluate_model(y_test, y_test_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    y_test_signal = np.sign(y_test)
    y_pred_signal = np.sign(y_test_pred)
    correct_predictions = (y_test_signal == y_pred_signal)
    directional_accuracy = correct_predictions.mean()

    # Legger til Directional Accuracy i output:
    return rmse

# Vis hvilke features som er vektet
def show_feature_importance(model, X, top_n=8):
    # Sjekk om modellen er tre-basert (RandomForest/GB)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("Denne modellen støtter ikke feature importance.")
        return
    # Koble til kolonnenavn
    pairs = zip(X.columns, importances)
    # Sorter synkende
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    # Print de viktigste
    for name, val in sorted_pairs[:top_n]:
        print(f"{name}: {val:.3f}")

# ------------------------------------

# Gridsearch optimization

def run_grid_search(model, params, X_train, y_train):
    tss = TimeSeriesSplit(n_splits=3)

    # Create GridSearchCV
    gs = GridSearchCV(estimator=model, param_grid=params, cv=tss, n_jobs=5, scoring='neg_mean_squared_error')
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def tune_random_forest(X_train, y_train):
    params = {
        "n_estimators": [10, 20, 40, 75], # Testet med flere parametre for å finne best param verdier
        "max_depth": [2, 3]                 # Og fjernet de som ikke var best
    }
    base = RandomForestRegressor(random_state=1)

    return run_grid_search(base, params, X_train, y_train) # Ferdigtrent beste modell

def tune_gradient_boost(X_train, y_train):
    params = {
        "n_estimators": [5, 10, 75],
        "max_depth": [2, 4, 7, 9],
        "learning_rate": [0.01]
    }
    base = GradientBoostingRegressor(random_state=1)
    return run_grid_search(base, params, X_train, y_train)

