use_RF_tuning = False
use_GB_tuning = True


# Kall alle funksjonen i en egen funksjon i riktig rekkefølge
def solve(filepath):
    from read_files import read_files, clean_stock_data
    from features import make_features, drop_na_features, make_X_y, scale_features
    from train import split_data, train_random_forest, make_predictions, evaluate_model, tune_gradient_boost, show_feature_importance, tune_random_forest
    df = read_files(filepath)
    df = clean_stock_data(df)
    df = make_features(df)
    df = drop_na_features(df)
    X, y = make_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    if use_RF_tuning:
        model = tune_random_forest(X_train, y_train)
    elif use_GB_tuning:
        model = tune_gradient_boost(X_train, y_train)
    else:
        model = train_random_forest(X_train, y_train)
    y_pred = make_predictions(model, X_test)
    rmse = evaluate_model(y_test, y_pred)
    show_feature_importance(model, X_train)
    print(filepath, "RMSE", rmse)



if __name__ == "__main__":
    files = ["../Data/DnB Stock Price History.csv",
             "../Data/Equinor Stock Price History.csv",
             "../Data/Frontline Stock Price History.csv",
             "../Data/Kitron Stock Price History.csv",
             "../Data/Orkla Stock Price History.csv",
             "../Data/Storebrand Stock Price History.csv"]
    for f in files:
        solve(f)



