import os
import pandas as pd

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
def clean_stock_data(df):
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


