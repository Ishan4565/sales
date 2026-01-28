import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sales Forecast", page_icon="📊", layout="wide")

st.title("📊 Sales Forecasting Dashboard")
st.markdown("### AI-Powered Sales Predictions")

@st.cache_data
def generate_sample_data():
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    base_sales = 500
    trend = np.linspace(0, 200, len(dates))
    seasonality = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 50, len(dates))
    
    sales = base_sales + trend + seasonality + noise
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales.clip(min=0)
    })
    
    return df

@st.cache_resource
def train_model(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    
    X = df[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
    y = df['sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    
    return model, score

df = generate_sample_data()
model, accuracy = train_model(df)

with st.sidebar:
    st.header("📈 Model Info")
    st.metric("Model Type", "Random Forest")
    st.metric("Accuracy (R²)", f"{accuracy:.2%}")
    st.metric("Training Data", f"{len(df)} days")

tab1, tab2 = st.tabs(["📊 Forecast", "📈 Historical"])

with tab1:
    st.subheader("Future Sales Forecast")
    
    days_to_forecast = st.slider("Days to Forecast", 7, 90, 30)
    
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)]
    
    future_data = pd.DataFrame({
        'date': future_dates
    })
    
    future_data['year'] = future_data['date'].dt.year
    future_data['month'] = future_data['date'].dt.month
    future_data['day'] = future_data['date'].dt.day
    future_data['day_of_week'] = future_data['date'].dt.dayofweek
    future_data['day_of_year'] = future_data['date'].dt.dayofyear
    
    X_future = future_data[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
    predictions = model.predict(X_future)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'].tail(90),
        y=df['sales'].tail(90),
        mode='lines',
        name='Historical',
        line=dict(color='#1976D2', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#00C853', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Forecast", f"${sum(predictions):,.0f}")
    with col2:
        st.metric("Avg Daily", f"${np.mean(predictions):,.0f}")
    with col3:
        st.metric("Peak Day", f"${max(predictions):,.0f}")
    with col4:
        st.metric("Lowest Day", f"${min(predictions):,.0f}")
    
    st.markdown("---")
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': predictions
    })
    
    st.dataframe(
        forecast_df.style.format({'Predicted Sales': '${:,.2f}'}),
        use_container_width=True
    )

with tab2:
    st.subheader("Historical Sales Trend")
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Sales',
        line=dict(color='#1976D2', width=2)
    ))
    
    fig_hist.update_layout(
        title="Full Historical Data",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=400
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("#### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${df['sales'].sum():,.0f}")
    with col2:
        st.metric("Average Daily", f"${df['sales'].mean():,.0f}")
    with col3:
        st.metric("Highest Day", f"${df['sales'].max():,.0f}")
    with col4:
        st.metric("Lowest Day", f"${df['sales'].min():,.0f}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Sales Forecasting Dashboard | Demo Version</p>
        <p>Built with Python, Scikit-learn, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)
```

**Update `requirements.txt`:**
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
streamlit==1.29.0
plotly==5.18.0