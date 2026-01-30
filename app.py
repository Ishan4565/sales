import streamlit as st
import os
from universal_engine import train_universal_model, predict_and_log

st.set_page_config(page_title="Inventory Drift Monitor", layout="wide")

db_url = os.getenv("postgresql://inventory_drift_db_user:Xf9BpwHH8zNTqmjap0W1bCLXKd3kUzni@dpg-d5uarfiqcgvc73asnf80-a/inventory_drift_db")
if not db_url:
    st.error("⚠️ DATABASE_URL is missing in Render Environment Settings")
else:
    st.success(f"✅ Cloud Environment Variable detected (Starts with: {db_url[:10]}...)")

st.title("🧥 Global Inventory & Drift Monitor")

model = train_universal_model()

with st.sidebar:
    st.header("Settings")
    product = st.text_input("Product Category", "Jackets")
    season = st.radio("Select Current Season", ["Spring", "Summer", "Fall", "Winter"])

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Average Temp (°C)", value=15)
    past_sales = st.number_input("Last Month's Units", value=500)

with col2:
    promo = st.checkbox("Active Promotion")
    actual = st.number_input("Actual Sales (Today)", value=450)

if st.button("Run AI Analysis"):
    try:
        prediction, drift = predict_and_log(product, season, temp, int(promo), past_sales, actual, model)
        
        st.metric("Predicted Demand", f"{prediction:.0f} units")
        st.metric("Drift Score", f"{drift:.2%}")
        
        if drift > 0.2:
            st.warning("🚨 Critical Drift Detected! Model retraining recommended.")
        else:
            st.success("✅ Model is stable. Prediction aligns with market reality.")
            
    except Exception as e:
        st.error(f"Database Error: {e}")
