import streamlit as st
import pandas as pd
from universal_engine import train_universal_model, predict_and_log

st.set_page_config(page_title="Seasonal Inventory AI", layout="wide")
import os
import streamlit as st

# DEBUG: This will show in your app if the variable is found
db_check = os.getenv("postgresql://inventory_drift_db_user:Xf9BpwHH8zNTqmjap0W1bCLXKd3kUzni@dpg-d5uarfiqcgvc73asnf80-a.singapore-postgres.render.com/inventory_drift_db")

if db_check:
    st.write(f"✅ System found a database URL starting with: {db_check[:15]}...")
else:
    st.error("❌ System still thinks DATABASE_URL is empty. Check Render Settings!")

st.title("🧥 Global Inventory & Drift Monitor")
st.markdown("Enter product details to calculate real-time demand drift.")

model = train_universal_model()

with st.sidebar:
    st.header("Settings")
    product = st.text_input("Product Category", value="Jackets")
    season = st.radio("Select Current Season", ["Spring", "Summer", "Fall", "Winter"])

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Average Temp (°C)", value=15)
        past_sales = st.number_input("Last Month's Units", value=500)
    with col2:
        promo = st.checkbox("Active Promotion")
        actual = st.number_input("Actual Sales (Today)", value=450)
    
    submitted = st.button("Run AI Analysis")

if submitted:
    pred, drift = predict_and_log(product, season, temp, int(promo), past_sales, actual, model)
    
    st.divider()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("AI Predicted", f"{pred:.0f} units")
    m2.metric("Actual Sales", f"{actual}")
    m3.metric("Drift Score", f"{drift:.2%}", delta_color="inverse")
    
    if drift > 0.2:
        st.error(f"High Drift Detected! {product} is disconnected from {season} logic.")
    else:

        st.success(f"System Stable for {product} in {season}.")

