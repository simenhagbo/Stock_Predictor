import yfinance as yf

df = yf.download("AAPL", period="1mo", interval="1d", progress=False)

print(df.head())
print(len(df))