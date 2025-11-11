# Personal Machine Learning Project

This is a project that uses machine learning to predict future stock prices for six norwegian companies: DnB, Equinor, Frontline, Kitron, Orkla, Storebrand. I've found and retrieved historical data for each company from [Investing.com](https://www.investing.com/).
For each company, weekly historical data was collected from June 1, 2015, to October 26, 2025. 

----------

The current status for this project is that it is under development.

----------

The goal for this project is to create a manchine learning model that can be used to predict next week's stock price. 

----------

To run this on your own computer please do the following steps:
1. Clone repo
   ```bash
   git clone https://github.com/simenh/Personal_ML_Project.git

2. Install the necessary package
   pip install -r requirements.txt

3. Run mainfile
   python main.py

4. Project build up folder structure

## Folder structure
Personal_ML_Project/
│
├── Data/               # CSV-files with stockprices
├── Source/             # Python-filer for behandling og modellering
│   ├── main.py
│   ├── train.py
│   ├── features.py
│   └── read_files.py
└── README.md

----------

5. Results from run: 

DnB: RMSE: 0.023 og MAE: 0.016

Equinor: RMSE: 0.033 og MAE: 0.027 

Frontline: RMSE: 0.027 og MAE: 0.020

Kitron: RMSE: 0.084 og MAE: 0.052

Orkla: RMSE: 0.027 og MAE: 0.019

Storebrand: RMSE: 0.028 og MAE: 0.020

----------

6. Further work
- Add more data to improve model generalisation
- Experiment with new features and technical indicators
- Test with Neural Networks to compare with ensemble methods
- Implement Genetic algorithms for hyperparameter tuning
- Visualize predictions in a web interface

