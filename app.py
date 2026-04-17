import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime
@st.cache_resource
def load_assets():
    model = joblib.load('walmart_rf_model.pkl')
    feature_names = joblib.load('features.pkl') 
    engine = create_engine("sqlite:///walmart_data.db")
    return model, feature_names, engine

model, feature_names, engine = load_assets()

st.set_page_config(page_title="Walmart Insights Engine", layout="wide")
st.title("📊 Walmart Sales Predictor + Explainable AI")
st.sidebar.header("Input Parameters")

def get_input():
    store = st.sidebar.number_input(
        "Store ID", 1, 45, 1, 
        help="Walmart has 45 anonymized stores. Choose any ID between 1 and 45."
    )
    dept = st.sidebar.number_input(
        "Dept ID", 1, 99, 1, 
        help="Department IDs represent specific categories (anonymized). Range: 1-99."
    )
    
    default_date = datetime(2012, 10, 26)
    date = st.sidebar.date_input(
    "Forecast Date", 
    value=default_date, 
    min_value=datetime(2010, 1, 1), # Range yahan se shuru hogi
    max_value=datetime(2035, 12, 31) # Range yahan tak jayegi
)
    is_holiday = st.sidebar.selectbox("Is it a Holiday?", [0, 1], help="Select 1 if the date falls in a holiday week.")
    lag_1 = st.sidebar.number_input("Last Week's Sales ($)", value=15000)
    
    data = {
        'Store': store, 'Dept': dept, 'Size': 151315, 'IsHoliday': is_holiday,
        'Temperature': 60.0, 'Fuel_Price': 3.5, 'CPI': 211.0, 'Unemployment': 8.1,
        'Month': date.month, 'WeekOfYear': date.isocalendar()[1],
        'lag_1': lag_1, 'lag_2': lag_1 * 0.95, 'lag_4': lag_1 * 1.05
    }
    return pd.DataFrame([data]), date, store, dept

input_df, selected_date, store_id, dept_id = get_input()

query_check = f"""
    SELECT Temperature, Fuel_Price, CPI, Unemployment, Size 
    FROM gold_sales_data 
    WHERE Store={store_id} AND Dept={dept_id} AND Date='{selected_date}'
    LIMIT 1
"""
db_record = pd.read_sql(query_check, con=engine)

if not db_record.empty:
    input_df.at[0, 'Temperature'] = db_record['Temperature'].iloc[0]
    input_df.at[0, 'Fuel_Price'] = db_record['Fuel_Price'].iloc[0]
    input_df.at[0, 'CPI'] = db_record['CPI'].iloc[0]
    input_df.at[0, 'Unemployment'] = db_record['Unemployment'].iloc[0]
    input_df.at[0, 'Size'] = db_record['Size'].iloc[0]
    st.sidebar.success("✅ Using exact data from Database!")
else:
    st.sidebar.info("ℹ️ for year 2010 to october 2012 it uses walmart dataset values for prediction Otherwise  Using default estimates values for Future Date.")
if st.button("Predict & Explain"):
    # AI Prediction
    pred = model.predict(input_df[feature_names])[0]
    
    # Header Metrics
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Weekly Sales", f"${pred:,.2f}")
    
    # Fetch Historical Data for Line Chart
    query = f"SELECT Date, Weekly_Sales FROM gold_sales_data WHERE Store={store_id} AND Dept={dept_id} ORDER BY Date"
    hist_df = pd.read_sql(query, con=engine)
    
    if not hist_df.empty:
        st.subheader("📈 Actual vs. Predicted Trend")
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Weekly_Sales'], name='Actual Past Sales', line=dict(color='royalblue')))
        fig_trend.add_trace(go.Scatter(
            x=[pd.to_datetime(selected_date)], y=[pred], 
            mode='markers+text', name='AI Prediction',
            text=["FORECAST"], textposition="top center",
            marker=dict(color='red', size=12, symbol='star')
        ))
        st.plotly_chart(fig_trend, use_container_width='stretch')
    else:
        st.warning("No historical data found for this Store/Dept combination to show the trend.")
    st.divider()
    st.subheader("🔍 AI Reasoning (Why this prediction?)")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df[feature_names])
    fig_shap, ax = plt.subplots()
    shap.bar_plot(shap_values[0], feature_names=feature_names, max_display=10, show=False)
    st.pyplot(fig_shap)
    
    st.markdown("### 📝 Key Insights:")
    top_impact_idx = np.abs(shap_values[0]).argsort()[-3:][::-1]
    
    for idx in top_impact_idx:
        feature = feature_names[idx]
        impact = shap_values[0][idx]
        direction = "increased" if impact > 0 else "decreased"
        
        st.write(f"🔹 **{feature}** has **{direction}** the sales forecast. This factor is a primary driver for this specific result.")
    if input_df['IsHoliday'][0] == 1:
        st.info("💡 **Note:** The model has applied a holiday premium to this forecast based on historical seasonal peaks.")