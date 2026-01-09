---
layout: default
title: Final Report
nav_order: 3
---

# Introduction/Background

Previous work on predicting housing prices has shown that both linear models such as linear regression, ridge, and lasso and tree based models such as random forest and gradient boosted trees can be used to find patterns in housing data. Studies have found that tree based models often perform better because they can capture the effects of many different factors like housing costs, rent, jobs, and migration. Our project builds on this by using Zillow housing data together with census population data and job statistics to predict which cities in the United States are likely to grow in the future.
The primary dataset being utilized is through Zillow.com. The historical data of house prices, rent values, sale to listing ratio, new construction sales, new construction median values, and more is available dating back to 2018. The secondary dataset that will be used for training is from Census.gov. It displays the populations of ~2000 major cities across the US over the last 5 years. The historical Zillow data will be used to predict city growth and migration then validated with the Census data. Once validated, the current Zillow data can be used to predict future population and migration data in various cities across the US.

* **Zillow Housing Data:** [https://www.zillow.com/research/data/](https://www.zillow.com/research/data/)
* **Census Population Data:** [Census.gov Time Series](https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-cities-and-towns.html)
* **Bureau of Labor Statistics:** [https://www.bls.gov/oes/current/oessrcma.htm](https://www.bls.gov/oes/current/oessrcma.htm)

## Problem Definition

Urban growth prediction can provide insight into future economic opportunities and population density in major cities, benefiting policymakers, real estate investors, and city planners. 68% of the world population is projected to live in urban areas in the next 25 years, increasing over 10% from the current population spread [1]. Urbanization will lead to an increase in “mega cities” with 20 million+ inhabitants, whereas others will have severe population decline. By predicting which urban centers will have the most growth and understanding key trends, city planning can be sustainably developed and managed to meet the needs of the expanding population, including housing, transportation, and energy infrastructure. 
Predicting the growth of cities complex due to the interdependence between housing markets, labor trends, and demographic shifts. This project aims to use a data driven approach to predict which U.S cities are most likely to grow by integrating datasets with housing, job, and population indicators. 


---

# Methods

### 3.1 Data Sources
The project will use supervised learning to predict which U.S. cities are most likely to grow. Housing, labor market, and demographic indicators were used for US cities 2018-2024. The following sources were merged using a custom preprocessing pipeline.

From Zillow, we used home values, rent values, income-to-home value ratios, and sale-to-list ratios. We also included days on market, new construction share, home sizes, and rental percentages. From the U.S Census Bureau, the annual population per city from 2015-2020 and 2020-2024 was used. From the Bureau of Labor Services, the employment rates per city and income measures were used.
 

### 3.2 Data Processing Methods
To process the raw data from Zillow, Census, and BLS, a script was implemented to automatically detect ID columns, year columns, and population columns, creating a unique city tag “city_id” to merge across datasets. The target feature was defined as next year’s log population change. 

Feature engineering was used to produce features tailored for population growth forecastingAn L1 lag target per city and state level average were added to incorporate temporal change. For all numeric features with greater than 80% coverage, L1, L2, and L3 lags were created (homevalue, bls_income). Relative deltas for census, BLS, and Zillow features were created to capture year over year proportional change. 

To handle missing values, median imputation was applied with the sci-kit learn library to apply rolling statistics across 2-4 year windows. For example, moving averages smoothed short term fluctuations. Winsor caps were used to trim the extremes learned on training data and clipped these target outliers to stabilize regression on the 1st and 99th percentile of data. Caps are learned on training data and applied to test and forecasting data. 

Candidate features were ranked using mutual information and F-regression hybrid selector. The top 40 features are selected per horizon, and were used by all three models. 


### 3.3 Machine Learning Models Implemented
* **Ridge Regression (Linear Baseline Model):** Provided a linear baseline with L2 regularization to measure feature importance. 

* **Random Forest Regressor:** Captured nonlinear effects among lagged and delta features. 

* **Gradient Boosted Regression Trees:** Handled complex nonlinear relationships more efficiently than Random Forests along with tabular, sequential data.

Hyperparameters were tuned for each of the models and listed below: 

| Model | Hyperparameters |
| :--- | :--- |
| **Ridge Regression** | alpha ∈ [0.05, 0.1, 0.3, 1.0, 3.0] <br> fit_intercept ∈ [True, False] |
| **Random Forest Regressor** | n_estimators ∈ [400, 1200] <br> max_depth ∈ [None, 8, 10, 12] <br> min_samples_split ∈ [2, 20] <br> min_samples_leaf ∈ [1, 6] <br> max_features ∈ ['sqrt', 'log2', 0.5] <br> bootstrap = True |
| **Gradient Boosted Regression Tree** | learning_rate=0.05 <br> max_iter=800 <br> max_depth=4 <br> min_samples_leaf=20 <br> max_bins=64 <br> l2_regularization=0.2 <br> early_stopping=True |

### 3.4 Training/Testing Splits & Validation
The training data was defined through a temporal split: by training on data up to 2022, and testing on 2023 data (first horizon). 

Retrospective performance was measured with the following metrics on the testing data from 2023 (h=1):  RMSE, MAE, and R². To evaluate rankings of cities, the Spearman correlation was computed.  A zero baseline RMSE was calculated as a reference threshold to define pass/fail criteria , predicting on the testing data (2023) using the mean of the training data (prior to 2022). All models must have an RMSE < 0.00970 based on zero baseline to pass.  

### 3.5 Horizon Prediction
From this benchmark for performance, the best model was chosen by having the lowest testing RMSE from 2023 (h=1) .  With this chosen model, two additional horizons (h= 3, 5) were used to predict forward looking forecasts for future urban growth for years 2025 and 2027. 
---

# Results & Discussion

### 4.1 Test Metric Data & Visualizations (2023)

> ![Screenshot 2025-12-02 at 11 08 14 PM](https://github.gatech.edu/user-attachments/assets/db5d93c0-6808-4f36-839d-f3b68a4a819f)
>
> *Figure 1: Predicted vs. Actual Rank (dlog pop. growth) for GBRT testing data for 2023. Points generally follow a stronger diagonal trend than unfiltered data. Increased data and windsor caps reduced noise*

> ![Screenshot 2025-12-02 at 11 08 40 PM](https://github.gatech.edu/user-attachments/assets/974256c0-b72a-4ae2-b333-33c8be0f98ba)
> *Figure 2: RMSE, MAE, and R^2 for Ridge Regression, Random Forest, and Gradient Boost Regression Trees. Random Forest and GBRT have the highest R^2 values. GBRT has the lowest*

> ![Screenshot 2025-12-02 at 11 34 04 PM](https://github.gatech.edu/user-attachments/assets/15151af5-60ca-4b0c-af4b-596adecc1a69)


> *Figure 3. Top 25 features ranked by importance in the GBRT model for predicting city growth in 2023. Construction sales ranked the most important.*

**Table 1: Testing metrics from model testing from 2023 (h=1) data**

| Model | RMSE | MAE | R² | Spearman |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | 0.0085 | 0.0061 | -0.0427 | 0.4237 |
| **Random Forest** | 0.0080 | 0.0060 | 0.0680 | 0.4239 |
| **Gradient Boosted Random Tree** | 0.0077 | 0.0058 | 0.1302 | 0.5144 |

### 4.2 Testing Metric Analysis 

All models were better than the threshold produced by zero baseline, detailing an RMSE of below 0.00970.

Implementation of the Random Forest improved MAE and RMSE as it handled nonlinear relationships and feature interactions (lagged, relative delta) better than ridge regression. Median imputation and feature selection make this model robust to missing or noisy city data. Because the horizon is only 1 year, the limited historical context shows feature drift between the training data and the test data, reducing generalization. There is also feature drift between pre-2020 training data and post-2020 testing, as the pandemic was a confounding factor in feature patterns. Random Forest performed better than Ridge because it was able to capture some of the small variation in year to year population changes that the linear model could not.
The R² values vary across the three models because the target variance in the 2023 dataset is small, so even modest errors inflate MSE relative to variance and penalize the R² metric, leading Ridge to be negative while Random Forest and GBRT remain slightly positive. This is why Random Forest and GBRT achieve slightly positive R² values even though the overall signal in the 2023 data is extremely small.


**Summary:**
* GBRT is the strongest midterm model, with the lowest error and best ranking skill
* Random Forest is competitive, but slightly less accurate
* Ridge Regression struggles due to inability to model nonlinear effects
* Spearman rank correlation range (0.42 - 0.51) indicate moderate ordinal skill despite low population variance in 2023. 

#### 4.2.1 Ridge Regression
The Ridge Regression model served as a linear baseline for predicting city-level population growth. While it offered interpretability and helped identify key features, its predictive performance was the weakest among the three models. As shown in Table 1, Ridge produced the highest RMSE and MAE relative to both the Random Forest and Gradient Boosted Regression Trees, reflecting its inability to capture nonlinear interactions in the housing, labor, and demographic features. The model returned a negative R² value because the population changes in 2023 are almost identical across cities. When the true values barely vary, even small prediction errors look large compared to that tiny amount of variation. As a result, R² becomes negative even though the model’s RMSE and MAE are small.  In contrast, both RF and GBRT achieved positive R² values, with GBRT performing best overall. Despite these limitations, Ridge still maintained moderate ordinal skill (Spearman 0.4237), indicating some ability to preserve rank ordering of city growth. Overall, Ridge provided a useful linear benchmark but lacked the flexibility needed to model the complex factors of short-term urban growth.

#### 4.2.2 Random Forest 
The Random Forest model served as a more flexible baseline for predicting city level population growth. While it handled patterns that Ridge could not and captured more useful signals from the housing, labor, and population features, its performance was in the middle compared to the three models. As shown in Figure 2, Random Forest produced lower RMSE and MAE than Ridge, showing that it picked up some of the nonlinear relationships present in the data. The model also returned a positive R² value because it was able to explain more of the small amount of variation in the 2023 population changes. Since the true values barely differ across cities, even small mistakes can make R² look weak, but Random Forest still performed better than the linear model. In terms of ranking, it achieved a Spearman correlation of 0.4239, which is almost identical to Ridge and indicates moderate ability to sort cities from faster to slower growth. Overall, Random Forest provided a stronger and more flexible baseline than Ridge, but it was still not as accurate as the Gradient Boosted Regression Trees.

#### 4.2.3 Gradient Boosted Random Tree
Gradient Boosted Random Tree is a model that helps to capture non-linear growth that the other models might miss. Unlike Random Forest, which constructs independent trees in parallel, GBRT builds trees sequentially, ensuring each new tree focuses on the residuals from the previous one and always correcting them.  As detailed in the quantitative metrics table, the Gradient Boosted Random Tree performs much better than the other models; it has the lowest RMSE of 0.0077 and the lowest MAE of 0.0058. Unlike the ridge, which gives us a negative R^2 value due to the low variance in the 2023 data, the GBRT model shows that it can capture patterns that other models might have not caught and gives us a positive score. Overall, the high Spearman correlation and the RMSE and MAE values show that GBRT is an effective way to order cities from fastest-growing to shrinking, making it a reliable model for generating forecasts. 

### 4.3 Final Forecasted Prediction Analysis
After evaluation of the three models above, the model that performed the best (based on the lowest RMSE) was the Gradient Boosted Random Tree. The data up to 2023 was used to train this GBRT model, and population predictions were made for 2025 and 2027. GBRT performed better than the linear and ridge baselines because its tree-based structure can capture the nonlinear patterns among housing, labor, and migration variables that influence city-level growth. The five cities with the highest predicted growth rates are shown in Figure 4 below. The single city with the largest predicted growth by 2027 is North Port, Florida, with a projected cumulative growth of roughly 9%. Across all 5 cities, the projected increase grows significantly between 2025 and 2027, indicating strengthening underlying demographic drivers. These forecasts should be interpreted with caution, however, as they assume that the socioeconomic trends observed in the training data will remain stable.

> ![Screenshot 2025-12-02 at 11 09 44 PM](https://github.gatech.edu/user-attachments/assets/7c874762-cea6-427e-bbe5-1d68ecea7a54)
> *Figure 4 : Bar Chart & City Growth Over time for Top 5 Cities using Final Model Selection*

## Next Steps
* **Improve Multi-Year Forecasting:** Extend the model to predict multi-year horizons (Horizon 2 and Horizon 3) instead of only the following year growth.

* **Add Additional Migration Features:** Incorporate more city-level indicators like rental market pressure.

* **Strengthen Temporal Evaluation:** Perform rolling-origin validation across multiple years to test the stability of the model over time, especially before and after feature drift events (the pandemic).

* **Model Improvements:**Test additional models like XGBoost, LightGBM, and CatBoost, and try it with simple stacking or a mix of Ridge, RF, and GBRT to improve both point accuracy and rank ordering.

* **Enhance Interpretability and Outputs:** Use feature importance tools (SHAP) to visualize factors driving each prediction and generate uncertainty levels to make the predictions more practical.


---

## References

1.  D. K. M. A. W. Wimalaweera, G. R. I. E. R. G. H. R. D. D. S. D. Wimalarathna and R. G. L. A. S. H. K. S. G. P. Samarakoon, "A Comparative Study of Regression Models for Housing Price Prediction," Int. J. Res. Sci. Manag., vol. 8, no. 12, pp. 29–33, Dec. 2021. [Online]. Available: [ResearchGate](https://www.researchgate.net/publication/383112591_A_Comparative_Study_of_Regression_Models_for_Housing_Price_Prediction).
2.  G. K. Reddy and S. G. Konda, "Predicting House Prices Using Machine Learning Models," Int. J. Inf. Technol. Eng., vol. 2, no. 1, pp. 1-6, Mar. 2024. [Online]. Available: [ResearchGate](https://www.researchgate.net/publication/394866214_Predicting_House_Prices_Using_Machine_Learning_Models).
3.  United Nations, “World population projected to live in urban areas by 2050, says UN,” DESA, [Online]. Available: [UN.org](https://www.un.org/uk/desa/68-world-population-projected-live-urban-areas-2050-says-un). [Accessed: Oct. 02, 2025].
4.  V. P. Didenko, "Traditional or advanced machine learning approaches? Which one is better for housing price prediction and uncertainty risk reduction," J. Econ. Dev., Environ., People, vol. 12, no. 2, pp. 28-36, 2023. [Online]. Available: [Virtus Interpress](https://virtusinterpress.org/Traditional-or-advanced-machine-learning-approaches-Which-one-is-better-for-housing-price-prediction-and-uncertainty-risk-reduction.html).

---

## Gantt Chart
![Screenshot 2025-12-02 at 11 25 39 PM](https://github.gatech.edu/user-attachments/assets/c86361c8-7029-4b4a-a2e8-60f1aeeae894)




---

## Contribution Table

| Name | Proposal Contributions |
| :--- | :--- |
| **Samit Khinvasara** | Regression Random Forest, and visualization |
| **Tanmay Joshi** | Data Processing and Cleaning, Git management,  GBRT |
| **Rishika Aila** | Initial Model Implementation |
| **Devasena Sitaram** | Report Writing, Future Prediction Analysis |
| **Kandhan Nadarajah** | Data Processing & Cleaning, Ridge Regression Analysis, Top 5 Cities Analysis |
