from read_files import read_files, clean_stock_data_csv, clean_yf_data, retrieve_data_from_yf
from features import make_features, drop_na_features, make_X_y, scale_features
from train import split_data, train_random_forest, make_predictions, evaluate_model, tune_gradient_boost, show_feature_importance, tune_random_forest
import os
import warnings
import time
# Ignorer advarsler om manglende feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------- API Data ------------------

use_RF_tuning_api = False
use_GB_tuning_api = True

def solve_api(tickers):
    # Importer data
    print("Henter ferske data fra Yahoo Finance...", end="\r")
    time.sleep(2)
    print("Data hentet! Starter analysen...   ")

    all_data = retrieve_data_from_yf(tickers)

    # Lagre data til csv som vi kan bruke i csv delen
    print("\nLagrer oppdaterte filer til Data mappen...")

    # Juster stien
    data_folder = "../Data"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for ticker, df in all_data.items():
        # Fjerner .OL for penere filnavn
        clean_name = ticker.replace(".OL", "")

        # Sett navneformatet for alle filene
        file_path = f"{data_folder}/{clean_name} Historiske Aksjepriser.csv"

        # Lagre til disk
        df.to_csv(file_path)
        print(f"Lagret {clean_name} Historiske Aksje Priser.csv")

    # Analyser
    print(f"\n{'Ticker':<12} | {'RMSE':<8} | {'Prediksjon for neste uke':<20}")
    print("-" * 50)
    # Iterer gjennom dataene
    for ticker, df in all_data.items():
        clean_name = ticker.replace(".OL", "")
        # Sett nye features
        df = make_features(df)
        df = drop_na_features(df)

        # Lag X og y
        X_all, y_all = make_X_y(df)
        X_all = scale_features(X_all)
        # Ta ut siste rad som skal brukes til "fremtidig" prediksjon
        siste_rad = X_all.iloc[-1].values.reshape(1, -1)  # samme features som modellen er trent på
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

        # 3. Spå fremtiden (neste uke)
        next_pred_array = make_predictions(model, siste_rad)
        next_pred = next_pred_array[0]  # Hent ut verdien fra arrayet

        # Formater utskriften (OPP/NED)
        retning = "OPP" if next_pred > 0 else "NED"
        prosent = next_pred * 100

        # Print resultat på en linje
        print(f"{clean_name:<12} | {rmse:.4f}   | {prosent:>.2f}% ({retning})")


if __name__ == "__main__":
    # Liste over aksjer du vil sjekke
    selskap_tickers = [
        "KIT.OL", # Kitron
        "STB.OL",  # Storebrand
        "MOWI.OL", # Mowi
        "FRO.OL",  # Frontline
        "ATEA.OL", # Atea
        "NORCO.OL", # Norconsult
        "LINK.OL" # Link Mobility
    ]

    solve_api(selskap_tickers)


### -------------- CSV DATA ------------------
import glob


use_RF_tuning = True
use_GB_tuning = False

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
    aksje_navn = filnavn.replace(" Historiske Aksjepriser.csv", "")

    # Print ytelse og fremtidig verdi
    print(f"{aksje_navn} | RMSE er: {rmse:.4f}")
    print(f"Predikert endring neste uke er: {neste_uke:.2%} ({retning})")



# if __name__ == "__main__":
#     files = glob.glob("../Data/*Historiske Aksjepriser.csv")
#     for f in files:
#         print("-" * 30)
#         solve_csv(f)
#         print("-" * 30)


