import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sales Forecast", page_icon="📊", layout="wide")

# --- DATABASE CONNECTION & MODEL LOADING ---
@st.cache_resource
def load_resources():
    # 1. Get URL from Render Environment Variables
    # If running locally, it falls back to your local connection
    db_url = os.environ.get('postgresql://sales_db_od4k_user:b1EyjxdHaTIj8B6fQjLbSVcK9zoOgtM0@dpg-d5svgfffte5s73cm0q5g-a/sales_db_od4k')
    
    if not db_url:
        # Fallback for local testing only
        db_url = "postgresql+psycopg2://postgres:ishan123@localhost:5432/postgres"
    
    # Render fix: SQLAlchemy requires 'postgresql://' instead of 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    engine = create_engine(db_url)
    
    # 2. Load pre-trained models (Ensure these are uploaded to GitHub)
    try:
        model = joblib.load('sales_forecast_model.pkl')
        scaler = joblib.load('sales_forecast_scaler.pkl')
        metadata = joblib.load('sales_forecast_metadata.pkl')
    except FileNotFoundError:
        st.error("Model files not found! Ensure .pkl files are in the repository.")
        return engine, None, None, None
        
    return engine, model, scaler, metadata

engine, model, scaler, metadata = load_resources()

# --- APP UI ---
st.title("📊 Sales Forecasting Dashboard")

if metadata:
    with st.sidebar:
        st.header("📈 Model Info")
        st.metric("Model Type", metadata['model_name'])
        st.metric("Accuracy (R²)", f"{metadata['metrics']['r2']:.2%}")
        st.info(f"Last Trained: {metadata['training_date']}")

    tab1, tab2 = st.tabs(["📊 Forecast View", "📈 Historical Analysis"])

    with tab1:
        st.subheader("Predicted Future Sales")
        # Pulling the forecast table you created in your training script
        try:
            forecast_df = pd.read_sql("SELECT * FROM sales_forecast ORDER BY forecast_date", engine)
            if not forecast_df.empty:
                fig = px.line(forecast_df, x='forecast_date', y='predicted_sales', 
                              title="30-Day Outlook", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(forecast_df)
        except Exception as e:
            st.warning("No forecast data found in database. Run training script first.")

    with tab2:
        st.subheader("Historical Sales Performance")
        query = """
        SELECT DATE_TRUNC('day', sale_date) as date, SUM(total_amount) as sales
        FROM sales GROUP BY 1 ORDER BY 1
        """
        try:
            hist_df = pd.read_sql(query, engine)
            st.line_chart(hist_df.set_index('date'))
        except:
            st.error("Could not load historical data. Check if 'sales' table exists.")
else:
    st.warning("App is waiting for model files and database connection...")
