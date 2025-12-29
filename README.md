# FUTURE_ML_01
Task 1:📈 Sales Forecasting with Prophet + Regressors
This project builds a weekly sales forecasting pipeline using Facebook Prophet with business regressors such as Discount, Quantity, Shipping Mode, and Region. Forecasts are generated separately for Furniture, Office Supplies, and Technology categories, and evaluated using MAE, RMSE, WAPE, and SMAPE.
🔧 Tech Stack
Python · Prophet · Pandas · NumPy · Scikit-Learn · Power BI
🚀 Workflow
Load & clean data (Sample - Superstore.csv)
One-hot encode categorical fields
Cap outliers at the 99th percentile
Aggregate sales weekly per category
Train Prophet models with regressors
Evaluate on the last 12 weeks
Export metrics & forecasts for Power BI
📊 Key Insight
Regressors improved accuracy — stable categories forecast better, while Technology remains more volatile.
▶ Visuals Created
Actual vs Forecast
Residual trends
Confidence intervals
Category metric comparison
🧠 Learnings
Feature engineering + regressors = stronger business-aware forecasts.
📌 Next Steps
Try new regressors
Tune Prophet parameters
Compare with ML models

Feel free to ⭐ the repo or suggest improvements!
