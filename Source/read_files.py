import os
import pandas as pd
import yfinance as yf
import time


# Use of Yahoo Finance API
def clean_yf_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.reset_index(inplace=True) # Gjør 'Date' til kolonne
    df["Date"] = pd.to_datetime(df["Date"]) # Setter riktig datoformat
    df.sort_values("Date", inplace=True) # Sorterer på stigende dato
    df = df.ffill() # Fyller hull i dataen med siste verdi
    return df

def retrieve_data_from_yf(tickers):
    all_data = {}
    # Gjennomgå tickers listen
    for t in tickers:
        print(f"Retrieving data for {t}.")
        # Prøv å hent data fra 6. juni 2015 til 01.12.2025, med 1 ukes intervall
        try:
            df = yf.download(t, start="2015-12-01", end="2025-12-01", interval="1wk", auto_adjust=False, progress=False)
            if df.empty:
                continue
            # Rens dataene
            df = clean_yf_data(df)

            df.rename(columns={"Close": "Price", "Volume": "Vol."}, inplace=True)
            # Lag en feature 'Change %'
            df["Change %"] = df["Price"].pct_change()
            df.set_index("Date", inplace=True)
            all_data[t] = df

            # Liten debug: vis de 3 første radene
            print(f"{t}: hentet {len(df)} rader")
            print(df.head(3))

        except Exception as e:
            print(f"Feil ved henting av {t}: {e}")
            continue
        time.sleep(8)
        print(f"Ferdig med {t} \n")
    return all_data


#  Use of csv files


# Lag en funksjon som leser inn filene
def read_files(filepath):
    return pd.read_csv(filepath)

def list_csv_files(mappe):
    csv_filer = []
    filnavn = os.listdir(mappe)
    for fil in filnavn:
        if fil.endswith(".csv"):
            csv_filer.append(fil)
    return csv_filer

# Lag en rense funksjon som renser kolonner og setter date til datetime
def clean_stock_data_csv(df):
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.ffill()

    # Rens 'Change%' kolonnen til desimal
    if "Change %" in df.columns:
        df['Change %'] = (
        df["Change %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        )
        df["Change %"] = pd.to_numeric(df["Change %"], errors="coerce") / 100

    # Rens 'Vol.' for 'M' og tekst
    if "Vol." in df.columns:
        def convert_volume(val):
            if pd.isna(val):
                return None
            val = str(val).replace(",", "").strip().upper()
            multiplier = 1
            if "K" in val:
                multiplier = 1_000
                val = val.replace("K", "")
            elif "M" in val:
                multiplier = 1_000_000
                val = val.replace("M", "")
            elif "B" in val:
                multiplier = 1_000_000_000
                val = val.replace("B", "")
            try:
                return float(val) * multiplier
            except:
                return None
        df["Vol."] = df["Vol."].apply(convert_volume)

    return df


