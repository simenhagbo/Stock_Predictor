# Personal Machine Learning Project

This is a project that uses machine learning to predict future stock prices. I have greated a script that shows multiple models of choice. Its running one function with data directly from the API, and one with csv files, which is retrieved and stored from the API. Easy to choose between the models: Random Forest, Tuned Random Forest, tuned Gradient booster. I've retrieved tickers from yfinance API and storing the ticker data as csv files for use of the csv version. 
For each ticker, weekly historical data was collected from June 1, 2015, to todays date. 

----------
## Status
The current status for this project is that it is under development.

----------
## Goal
The goal for this project is to create a manchine learning model that can be used to predict next week's stock price. 

----------
## How to run
To run this on your own computer please do the following steps:
1. Clone repo
   ```bash
   git clone https://github.com/simenh/Personal_ML_Project.git

2. Install the necessary package
   ```bash
   pip install -r requirements.txt

4. Run mainfile
   ```bash 
   python main.py

6. Project build up folder structure

## Folder structure
Personal_ML_Project/
│
├── Data/               # CSV-files with stockprices
├── Source/             # Python-files
│   ├── main.py
│   ├── train.py
│   ├── features.py
│   └── read_files.py
└── README.md

----------

## Results from run 
### API Tuned Gradient Booster
Ticker       | RMSE     | Prediction for next week
--------------------------------------------------
KIT          | 0.0551   | 0.70% (UP)
STB          | 0.0274   | 0.38% (UP)
MOWI         | 0.0281   | -0.31% (DOWN)
FRO          | 0.0600   | 0.38% (UP)
ATEA         | 0.0298   | 0.30% (UP)
NORCO        | 0.0291   | 0.64% (UP)
LINK         | 0.0437   | -0.25% (DOWN)

----------

## Further work
- Experiment with new features and technical indicators
- Test with Neural Networks to compare with ensemble methods
- Implement Genetic algorithms for hyperparameter tuning
- Visualize predictions in a web interface

