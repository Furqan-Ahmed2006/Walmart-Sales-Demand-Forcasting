# Walmart-Sales-Demand-Forcasting
An end-to-end Demand Forecasting and Time Series Analysis application that predicts weekly sales for Walmart stores. This project combines machine learning with Explainable AI (XAI) to provide transparent and actionable business insights.

🚀 Key Features
Demand Forecasting: Predicts weekly sales based on historical patterns, store size, and seasonal factors.

Time Series Integration: Uses temporal features like Lag_1, WeekOfYear, and Month to capture trends.

Hybrid Data Sourcing: - Automatically fetches actual historical data from an integrated SQLite database for past dates (2010-2012).

Simulates future forecasts (up to 2026) using user-defined proxy parameters.

Explainable AI (SHAP): Uses SHAP values to explain why the model made a specific prediction, highlighting the top drivers of sales.

Interactive Dashboard: Built with Streamlit, featuring Plotly line charts for "Actual vs. Predicted" trend analysis.

🏗️ Technical Architecture
The project follows a modular data science workflow:

ETL Pipeline: Raw data was cleaned and processed using MySQL.

Feature Engineering: Created lag features and seasonal indicators to transform the problem into a supervised learning task.

Model: A Random Forest Regressor was trained and optimized for Mean Absolute Error (MAE).

Portability: For deployment, the database was migrated to SQLite to ensure the app is serverless and easy to run.

🛠️ Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, Joblib

Visualization: Plotly, Matplotlib, SHAP (XAI)

Database: MySQL (ETL), SQLite (Deployment)

Deployment: Huggingface Spaces

🔗https://huggingface.co/spaces/Furqan2006/walmart_sales_insight

In retail, understocking leads to lost revenue, while overstocking increases holding costs. This engine solves that by providing data-driven forecasts. Moreover, by adding SHAP explanations, we build trust with business stakeholders by showing them exactly which factors (like Holidays or Last Week's performance) are impacting the numbers.
