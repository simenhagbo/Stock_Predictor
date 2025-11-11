import numpy as np
import pandas as pd

def add_market_reaction(df):
    vol_factor = np.log10(df["Vol."] / df["Vol."].mean())
    df["Market_Reaction"] = df["Change %"] * vol_factor
    return df

# Lager en funksjon som lager nye features
def make_features(df):
    #df["SMA5"] = df["Price"].rolling(5).mean()
    #df["SMA20"] = df["Price"].rolling(20).mean()
    df["momentum_3w"] = df["Price"] / df["Price"].shift(3) - 1 # Endring siste 3 uker - viser trendstyrke
    df["prev_change"] = df["Change %"].shift(1)
    df["volatility_5w"] = df["Change %"].rolling(window=5).std()
    df = add_market_reaction(df)
    return df

def drop_na_features(df):
    df = df.dropna(subset=["prev_change", "volatility_5w", "momentum_3w"])
    return df

# Lager X og y for videre bruk
def make_X_y(df):
    X = df.drop("Change %", axis=1).copy()
    y = df["Change %"].shift(-1).rename("TargetChange") # Neste ukes pris

    # Slår sammen X og y midlertidig og fjerner tomme rader
    data = pd.concat([X, y], axis=1).dropna()

    # Deler opp igjen etterpå
    X = data.drop("TargetChange", axis=1)
    y = data["TargetChange"]

    return X, y

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled


#if __name__ == "__main__":


