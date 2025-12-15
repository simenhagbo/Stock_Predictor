from read_files import read_files, clean_stock_data_csv, clean_yf_data, retrieve_data_from_yf
from features import make_features, drop_na_features, make_X_y, scale_features
from train import split_data, train_random_forest, make_predictions, evaluate_model, tune_gradient_boost, show_feature_importance, tune_random_forest
import os
import warnings
# Ignorer advarsler om manglende feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------- API Data ------------------

use_RF_tuning_api = False
use_GB_tuning_api = False

def solve_api(tickers):
    # Importer data
    all_data = retrieve_data_from_yf(tickers)

    # Iterer gjennom dataene
    for ticker, df in all_data.items():
        print(f"Analyserer {ticker}..")
        # Sett nye features
        df = make_features(df)
        df = drop_na_features(df)

        # Lag X og y
        X_all, y_all = make_X_y(df)
        # Ta ut siste rad som skal brukes til "fremtidig" prediksjon
        siste_rad = X_all.iloc[-1]  # samme features som modellen er trent på
        X = X_all.iloc[:-1]  # alt unntatt siste rad til trening/testing
        y = y_all.iloc[:-1]

        # Splitt i trening og test
        X_train, X_test, y_train, y_test = split_data(X, y)
        # Tren modellen
        if use_RF_tuning_api:
            model = tune_random_forest(X_train, y_train)
        elif use_GB_tuning_api:
            model = tune_gradient_boost(X_train, y_train)
        else:
            model = train_random_forest(X_train, y_train)
        y_pred = make_predictions(model, X_test)
        rmse = evaluate_model(y_test, y_pred)

        # Bruk modellen på siste observasjon for å spå neste uke
        next_pred = make_predictions(model, siste_rad.to_frame().T)
        print(f"Modellen spår at prisen endres med: {next_pred[0]:.4f}")


selskap_tickers = []


### -------------- CSV DATA ------------------



use_RF_tuning = False
use_GB_tuning = True

# Kall alle funksjonen i en egen funksjon i riktig rekkefølge
def solve_csv(filepath):
    # Les inn og rens data
    df = read_files(filepath)
    df = clean_stock_data_csv(df)

    # Lag features
    df = make_features(df)
    df = drop_na_features(df)

    # Lag X og y
    X, y = make_X_y(df)

    # Skaler X
    X = scale_features(X)

    # Splitt, tren og evaluer
    X_train, X_test, y_train, y_test = split_data(X, y)
    if use_RF_tuning:
        model = tune_random_forest(X_train, y_train)
    elif use_GB_tuning:
        model = tune_gradient_boost(X_train, y_train)
    else:
        model = train_random_forest(X_train, y_train)
    y_pred = make_predictions(model, X_test)
    rmse = evaluate_model(y_test, y_pred)

    # Siste ukes data
    siste_data = X.iloc[-1].values

    # reshape til (1, antall_features) fordi modellen forventer en batch
    siste_data_reshape = siste_data.reshape(1, -1)

    # Prediker fremtiden
    neste_uke = model.predict(siste_data_reshape)[0]

    retning = "OPP" if neste_uke > 0 else "NED"

    # Trekk ut filnavnet
    filnavn = os.path.basename(filepath)

    # Fjerner overflødig tekst så vi sitter igjen med selskapsnavn
    aksje_navn = filnavn.replace(" Stock Price History.csv", "")

    # Print ytelse og fremtidig verdi
    print(f"{aksje_navn} | RMSE er: {rmse:.4f}")
    print(f"Predikert endring neste uke er: {neste_uke:.2%} ({retning})")



if __name__ == "__main__":
    files = ["../Data/DnB Stock Price History.csv",
             "../Data/Equinor Stock Price History.csv",
             "../Data/Frontline Stock Price History.csv",
             "../Data/Kitron Stock Price History.csv",
             "../Data/Orkla Stock Price History.csv",
             "../Data/Storebrand Stock Price History.csv"]
    for f in files:
        print("-" * 30)
        solve_csv(f)
        print("-" * 30)


