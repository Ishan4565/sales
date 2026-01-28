import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Sales Forecast", page_icon="📊", layout="wide")

@st.cache_resource
def load_model_and_connect():
    password = 'ishan123'
    encoded_password = quote_plus(password)
    engine = create_engine(f'postgresql+psycopg2://postgres:{encoded_password}@localhost:5432/postgres')
    
    model = joblib.load('sales_forecast_model.pkl')
    scaler = joblib.load('sales_forecast_scaler.pkl')
    metadata = joblib.load('sales_forecast_metadata.pkl')
    
    return engine, model, scaler, metadata

engine, model, scaler, metadata = load_model_and_connect()

st.title("📊 Sales Forecasting Dashboard")
st.markdown("### AI-Powered Sales Predictions")

with st.sidebar:
    st.header("📈 Model Info")
    st.metric("Model Type", metadata['model_name'])
    st.metric("Accuracy (R²)", f"{metadata['metrics']['r2']:.2%}")
    st.metric("Avg Error", f"${metadata['metrics']['mae']:.2f}")
    
    st.markdown("---")
    st.header("📅 Forecast Settings")
    forecast_days = st.slider("Days to Forecast", 7, 90, 30)
    
    st.markdown("---")
    st.info(f"Model trained on: {metadata['training_date']}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Historical", "🎯 Accuracy", "💡 Insights"])

with tab1:
    st.subheader("Future Sales Forecast")
    
    query = "SELECT * FROM sales_forecast ORDER BY forecast_date"
    forecast_df = pd.read_sql(query, engine)
    
    if len(forecast_df) > 0:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['predicted_sales'],
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='#00C853', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="30-Day Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Sales ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_forecast = forecast_df['predicted_sales'].sum()
            st.metric("Total Forecasted", f"${total_forecast:,.0f}")
        
        with col2:
            avg_daily = forecast_df['predicted_sales'].mean()
            st.metric("Avg Daily Sales", f"${avg_daily:,.0f}")
        
        with col3:
            max_day = forecast_df['predicted_sales'].max()
            st.metric("Peak Day", f"${max_day:,.0f}")
        
        with col4:
            min_day = forecast_df['predicted_sales'].min()
            st.metric("Lowest Day", f"${min_day:,.0f}")
        
        st.markdown("---")
        st.markdown("#### 📋 Detailed Forecast")
        st.dataframe(
            forecast_df.style.format({
                'predicted_sales': '${:,.2f}'
            }),
            use_container_width=True
        )
        
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Historical Sales Trends")
    
    query = """
    SELECT 
        DATE_TRUNC('day', sale_date) as date,
        SUM(total_amount) as daily_sales,
        COUNT(*) as num_transactions
    FROM sales
    GROUP BY date
    ORDER BY date
    """
    
    historical_df = pd.read_sql(query, engine)
    
    fig = px.line(
        historical_df,
        x='date',
        y='daily_sales',
        title='Historical Daily Sales',
        labels={'daily_sales': 'Sales ($)', 'date': 'Date'}
    )
    
    fig.update_traces(line_color='#1976D2', line_width=2)
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        query_category = """
        SELECT 
            p.category,
            SUM(s.total_amount) as total_sales
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.category
        ORDER BY total_sales DESC
        """
        
        category_df = pd.read_sql(query_category, engine)
        
        fig_cat = px.pie(
            category_df,
            values='total_sales',
            names='category',
            title='Sales by Category'
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        query_region = """
        SELECT 
            c.region,
            SUM(s.total_amount) as total_sales
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        GROUP BY c.region
        ORDER BY total_sales DESC
        """
        
        region_df = pd.read_sql(query_region, engine)
        
        fig_region = px.bar(
            region_df,
            x='region',
            y='total_sales',
            title='Sales by Region',
            color='total_sales',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig_region, use_container_width=True)

with tab3:
    st.subheader("Model Accuracy Metrics")
    
    query = "SELECT * FROM model_performance LIMIT 100"
    perf_df = pd.read_sql(query, engine)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_actual_vs_pred = go.Figure()
        
        fig_actual_vs_pred.add_trace(go.Scatter(
            x=perf_df['actual_sales'],
            y=perf_df['predicted_sales'],
            mode='markers',
            name='Predictions',
            marker=dict(size=8, color='#2196F3', opacity=0.6)
        ))
        
        min_val = min(perf_df['actual_sales'].min(), perf_df['predicted_sales'].min())
        max_val = max(perf_df['actual_sales'].max(), perf_df['predicted_sales'].max())
        
        fig_actual_vs_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_actual_vs_pred.update_layout(
            title='Actual vs Predicted Sales',
            xaxis_title='Actual Sales ($)',
            yaxis_title='Predicted Sales ($)',
            height=400
        )
        
        st.plotly_chart(fig_actual_vs_pred, use_container_width=True)
    
    with col2:
        fig_error = px.histogram(
            perf_df,
            x='percentage_error',
            title='Prediction Error Distribution',
            labels={'percentage_error': 'Error (%)'},
            nbins=30
        )
        
        fig_error.update_layout(height=400)
        st.plotly_chart(fig_error, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📊 Performance Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae = abs(perf_df['difference']).mean()
        st.metric("Mean Absolute Error", f"${mae:,.2f}")
    
    with col2:
        rmse = np.sqrt((perf_df['difference'] ** 2).mean())
        st.metric("RMSE", f"${rmse:,.2f}")
    
    with col3:
        mape = abs(perf_df['percentage_error']).mean()
        st.metric("Avg % Error", f"{mape:.2f}%")
    
    with col4:
        accuracy = (1 - abs(perf_df['percentage_error']).mean() / 100) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")

with tab4:
    st.subheader("💡 Business Insights")
    
    query_insights = """
    SELECT 
        p.product_name,
        p.category,
        COUNT(s.sale_id) as num_sales,
        SUM(s.total_amount) as total_revenue,
        AVG(s.total_amount) as avg_sale_value
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_name, p.category
    ORDER BY total_revenue DESC
    LIMIT 10
    """
    
    top_products = pd.read_sql(query_insights, engine)
    
    st.markdown("#### 🏆 Top 10 Products by Revenue")
    st.dataframe(
        top_products.style.format({
            'total_revenue': '${:,.2f}',
            'avg_sale_value': '${:,.2f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Growth Opportunities")
        st.info("""
        **Recommendations:**
        - Focus on high-performing categories
        - Increase inventory for top products
        - Target regions with growth potential
        - Optimize pricing for best sellers
        """)
    
    with col2:
        st.markdown("#### ⚠️ Risk Factors")
        st.warning("""
        **Watch Out For:**
        - Seasonal fluctuations
        - Market competition
        - Supply chain issues
        - Customer behavior changes
        """)
    
    st.markdown("---")
    
    query_monthly = """
    SELECT 
        DATE_TRUNC('month', sale_date) as month,
        SUM(total_amount) as monthly_sales
    FROM sales
    GROUP BY month
    ORDER BY month
    """
    
    monthly_df = pd.read_sql(query_monthly, engine)
    
    fig_monthly = go.Figure()
    
    fig_monthly.add_trace(go.Bar(
        x=monthly_df['month'],
        y=monthly_df['monthly_sales'],
        name='Monthly Sales',
        marker_color='#4CAF50'
    ))
    
    fig_monthly.update_layout(
        title='Monthly Sales Trend',
        xaxis_title='Month',
        yaxis_title='Sales ($)',
        height=400
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Sales Forecasting Dashboard | Powered by Machine Learning</p>
        <p>Built with Python, PostgreSQL, Scikit-learn, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)