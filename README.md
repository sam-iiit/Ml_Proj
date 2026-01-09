
# Problem: 
Identify which U.S. cities are likely to grow using signals from housing, jobs, and migration.

# Solution: 
Train supervised models on Zillow, Census, and BLS features to predict next year population growth, validate against Census, then generate ranked city forecasts.

## To run the Code 
1. Please Download Code.py, CodeCheck.py and CodeCheckFuture.py 

2. Run  python CodeCheck.py
    - Trains on 2018–2022, tests on 2023, horizon H=1.
    - Creates:
        - reports/midterm/figures/model_compare.png: RMSE, MAE, R² bar chart for ridge, RF, GBRT
        - reports/midterm/figures/rank_scatter_best.png: rank agreement for the best model
        - reports/midterm/models/test_predictions.csv: city_id, year, predicted vs true dlog and percent growth
        - reports/midterm/models/top_cities_predictions.csv: top 50 cities by predicted growth for 2023

3. Run python CodeCheckFuture.py
    - Fits on all available history, produces forward forecasts for horizons 1, 3, 5 from the last year in the dataset.
    - Creates:
        - reports/final/forecast_h1_rf.csv and _top50.csv
        - reports/final/forecast_h3_rf.csv and _top50.csv
        - reports/final/forecast_h5_ridge.csv and _top50.csv
        - reports/final/figures/rank_scatter_h1_rf.png
