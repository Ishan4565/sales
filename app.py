import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import json
import os

st.set_page_config(page_title="Inventory Drift Monitor", layout="wide", page_icon="🛍️")

@st.cache_data
def generate_historical_data(product_category, season, days=90):
    np.random.seed(42)
    
    season_multipliers = {
        'Winter': {'Jackets': 1.8, 'Gloves': 1.9, 'Boots': 1.7, 'Scarves': 1.6},
        'Summer': {'Jackets': 0.4, 'Gloves': 0.3, 'Boots': 0.5, 'Scarves': 0.3},
        'Spring': {'Jackets': 0.9, 'Gloves': 0.7, 'Boots': 0.8, 'Scarves': 0.7},
        'Fall': {'Jackets': 1.3, 'Gloves': 1.2, 'Boots': 1.2, 'Scarves': 1.1}
    }
    
    base_sales = {
        'Jackets': 150,
        'Gloves': 200,
        'Boots': 120,
        'Scarves': 180
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    base = base_sales.get(product_category, 100)
    multiplier = season_multipliers.get(season, {}).get(product_category, 1.0)
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        day_of_week = current_date.weekday()
        month = current_date.month
        
        is_weekend = day_of_week >= 5
        is_holiday = np.random.random() < 0.05
        has_promotion = np.random.random() < 0.2
        
        temp = np.random.normal(15 if season == 'Spring' else 25 if season == 'Summer' else 5 if season == 'Winter' else 12, 5)
        
        sales = base * multiplier
        
        if is_weekend:
            sales *= 1.2
        if is_holiday:
            sales *= 1.4
        if has_promotion:
            sales *= 1.3
        
        temp_factor = 1 + (15 - temp) * 0.02 if product_category in ['Jackets', 'Gloves', 'Boots', 'Scarves'] else 1
        sales *= temp_factor
        
        sales = max(0, int(sales + np.random.normal(0, sales * 0.15)))
        
        data.append({
            'date': current_date,
            'sales': sales,
            'temperature': temp,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'has_promotion': has_promotion,
            'month': month
        })
    
    return pd.DataFrame(data)

def calculate_drift_metrics(historical_sales, current_sales, historical_temp, current_temp):
    if len(historical_sales) < 10:
        return None
    
    hist_mean = np.mean(historical_sales)
    hist_std = np.std(historical_sales)
    
    z_score = (current_sales - hist_mean) / hist_std if hist_std > 0 else 0
    
    percentile = stats.percentileofscore(historical_sales, current_sales)
    
    ks_statistic, ks_pvalue = stats.ks_2samp(historical_sales, [current_sales] * 10)
    
    reference_counts, bin_edges = np.histogram(historical_sales, bins=10)
    current_bin = np.digitize([current_sales], bin_edges)[0] - 1
    current_bin = min(max(0, current_bin), 9)
    
    ref_percents = reference_counts / len(historical_sales)
    curr_percents = np.zeros(10)
    curr_percents[current_bin] = 1.0
    
    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
    
    psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
    
    drift_score = abs(psi) * 100
    
    temp_drift = abs(current_temp - np.mean(historical_temp)) / np.std(historical_temp) if np.std(historical_temp) > 0 else 0
    
    return {
        'z_score': z_score,
        'percentile': percentile,
        'psi': psi,
        'drift_score': drift_score,
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'historical_mean': hist_mean,
        'historical_std': hist_std,
        'temp_drift': temp_drift,
        'is_anomaly': abs(z_score) > 2 or drift_score > 20
    }

def train_forecasting_model(historical_df):
    X = historical_df[['temperature', 'day_of_week', 'is_weekend', 'is_holiday', 'has_promotion', 'month']].astype(float)
    y = historical_df['sales']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def predict_expected_sales(model, current_temp, day_of_week, is_weekend, is_holiday, has_promotion, month):
    features = np.array([[current_temp, day_of_week, is_weekend, is_holiday, has_promotion, month]])
    prediction = model.predict(features)[0]
    return max(0, int(prediction))

def main():
    st.title("🛍️ Global Inventory & Drift Monitor")
    st.markdown("### Real-time Sales Drift Detection & Forecasting")
    
    with st.sidebar:
        st.header("Settings")
        
        st.subheader("Product Category")
        product_category = st.selectbox(
            "Select Product",
            ["Jackets", "Gloves", "Boots", "Scarves"],
            label_visibility="collapsed"
        )
        
        st.subheader("Select Current Season")
        season = st.radio(
            "Season",
            ["Spring", "Summer", "Fall", "Winter"],
            label_visibility="collapsed"
        )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        avg_temp = st.number_input("Average Temp (°C)", min_value=-20, max_value=45, value=15, step=1)
    
    with col2:
        active_promotion = st.checkbox("Active Promotion", value=False)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        last_month_units = st.number_input("Last Month's Units", min_value=0, max_value=10000, value=500, step=10)
    
    with col4:
        actual_sales_today = st.number_input("Actual Sales (Today)", min_value=0, max_value=5000, value=450, step=10)
    
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Analyzing historical data and detecting drift..."):
            historical_df = generate_historical_data(product_category, season, days=90)
            
            current_date = datetime.now()
            day_of_week = current_date.weekday()
            is_weekend = day_of_week >= 5
            is_holiday = False
            month = current_date.month
            
            model, metrics = train_forecasting_model(historical_df)
            
            expected_sales = predict_expected_sales(
                model, avg_temp, day_of_week, is_weekend, is_holiday, active_promotion, month
            )
            
            drift_metrics = calculate_drift_metrics(
                historical_df['sales'].values,
                actual_sales_today,
                historical_df['temperature'].values,
                avg_temp
            )
            
            st.success("✅ Analysis Complete!")
            
            st.markdown("---")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Expected Sales",
                    f"{expected_sales:,}",
                    delta=None
                )
            
            with metric_col2:
                variance = ((actual_sales_today - expected_sales) / expected_sales * 100) if expected_sales > 0 else 0
                st.metric(
                    "Actual Sales",
                    f"{actual_sales_today:,}",
                    delta=f"{variance:+.1f}%",
                    delta_color="normal" if abs(variance) < 10 else "inverse"
                )
            
            with metric_col3:
                st.metric(
                    "Drift Score",
                    f"{drift_metrics['drift_score']:.1f}",
                    delta="Anomaly" if drift_metrics['is_anomaly'] else "Normal",
                    delta_color="inverse" if drift_metrics['is_anomaly'] else "normal"
                )
            
            with metric_col4:
                st.metric(
                    "Historical Avg",
                    f"{drift_metrics['historical_mean']:.0f}",
                    delta=f"±{drift_metrics['historical_std']:.0f}"
                )
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Drift Analysis", "📈 Historical Trends", "🎯 Model Performance", "📋 Detailed Metrics"])
            
            with tab1:
                st.subheader("Drift Detection Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    drift_status = "🔴 DRIFT DETECTED" if drift_metrics['is_anomaly'] else "🟢 NORMAL"
                    st.markdown(f"### {drift_status}")
                    
                    st.markdown(f"""
                    **Z-Score:** {drift_metrics['z_score']:.2f}  
                    **Percentile:** {drift_metrics['percentile']:.1f}%  
                    **PSI Score:** {drift_metrics['psi']:.4f}  
                    **Temperature Drift:** {drift_metrics['temp_drift']:.2f}σ
                    """)
                    
                    if drift_metrics['is_anomaly']:
                        st.warning("""
                        ⚠️ **Anomaly Detected!**
                        - Sales significantly different from historical patterns
                        - Recommend investigating external factors
                        - Consider adjusting inventory levels
                        """)
                    else:
                        st.info("""
                        ✅ **Sales Within Normal Range**
                        - Performance aligns with historical data
                        - No immediate action required
                        - Continue monitoring
                        """)
                
                with col2:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=historical_df['sales'],
                        name='Historical Distribution',
                        nbinsx=30,
                        marker_color='lightblue',
                        opacity=0.7
                    ))
                    
                    fig.add_vline(
                        x=actual_sales_today,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Today's Sales",
                        annotation_position="top"
                    )
                    
                    fig.add_vline(
                        x=drift_metrics['historical_mean'],
                        line_dash="dot",
                        line_color="green",
                        annotation_text="Historical Mean",
                        annotation_position="bottom"
                    )
                    
                    fig.update_layout(
                        title="Sales Distribution",
                        xaxis_title="Sales Units",
                        yaxis_title="Frequency",
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Historical Sales Trends")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['sales'],
                    mode='lines',
                    name='Historical Sales',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=[datetime.now()],
                    y=[actual_sales_today],
                    mode='markers',
                    name='Today',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                fig.add_hline(
                    y=drift_metrics['historical_mean'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Avg: {drift_metrics['historical_mean']:.0f}"
                )
                
                fig.add_hline(
                    y=drift_metrics['historical_mean'] + 2*drift_metrics['historical_std'],
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="+2σ"
                )
                
                fig.add_hline(
                    y=drift_metrics['historical_mean'] - 2*drift_metrics['historical_std'],
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="-2σ"
                )
                
                fig.update_layout(
                    title=f"{product_category} Sales Over Last 90 Days",
                    xaxis_title="Date",
                    yaxis_title="Sales Units",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    temp_fig = px.scatter(
                        historical_df,
                        x='temperature',
                        y='sales',
                        title='Sales vs Temperature',
                        trendline='ols',
                        labels={'temperature': 'Temperature (°C)', 'sales': 'Sales Units'}
                    )
                    st.plotly_chart(temp_fig, use_container_width=True)
                
                with col2:
                    day_avg = historical_df.groupby('day_of_week')['sales'].mean().reset_index()
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    day_avg['day_name'] = day_avg['day_of_week'].apply(lambda x: day_names[x])
                    
                    day_fig = px.bar(
                        day_avg,
                        x='day_name',
                        y='sales',
                        title='Average Sales by Day of Week',
                        labels={'day_name': 'Day', 'sales': 'Avg Sales'}
                    )
                    st.plotly_chart(day_fig, use_container_width=True)
            
            with tab3:
                st.subheader("ML Model Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
                
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                
                with col3:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                
                st.markdown("---")
                
                predictions = model.predict(historical_df[['temperature', 'day_of_week', 'is_weekend', 'is_holiday', 'has_promotion', 'month']].astype(float))
                
                pred_fig = go.Figure()
                
                pred_fig.add_trace(go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['sales'],
                    mode='markers',
                    name='Actual',
                    marker=dict(size=6, color='blue', opacity=0.6)
                ))
                
                pred_fig.add_trace(go.Scatter(
                    x=historical_df['date'],
                    y=predictions,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2)
                ))
                
                pred_fig.update_layout(
                    title='Actual vs Predicted Sales',
                    xaxis_title='Date',
                    yaxis_title='Sales Units',
                    height=400
                )
                
                st.plotly_chart(pred_fig, use_container_width=True)
                
                feature_importance = pd.DataFrame({
                    'Feature': ['Temperature', 'Day of Week', 'Weekend', 'Holiday', 'Promotion', 'Month'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                imp_fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(imp_fig, use_container_width=True)
            
            with tab4:
                st.subheader("Detailed Metrics")
                
                metrics_data = {
                    'Metric': [
                        'Expected Sales',
                        'Actual Sales',
                        'Variance (%)',
                        'Historical Mean',
                        'Historical Std Dev',
                        'Z-Score',
                        'Percentile',
                        'PSI Score',
                        'Drift Score',
                        'Temperature Drift',
                        'KS Statistic',
                        'KS P-Value',
                        'Anomaly Status'
                    ],
                    'Value': [
                        f"{expected_sales:,}",
                        f"{actual_sales_today:,}",
                        f"{((actual_sales_today - expected_sales) / expected_sales * 100):.2f}%" if expected_sales > 0 else "N/A",
                        f"{drift_metrics['historical_mean']:.2f}",
                        f"{drift_metrics['historical_std']:.2f}",
                        f"{drift_metrics['z_score']:.2f}",
                        f"{drift_metrics['percentile']:.2f}%",
                        f"{drift_metrics['psi']:.4f}",
                        f"{drift_metrics['drift_score']:.2f}",
                        f"{drift_metrics['temp_drift']:.2f}σ",
                        f"{drift_metrics['ks_statistic']:.4f}",
                        f"{drift_metrics['ks_pvalue']:.4f}",
                        "🔴 Anomaly" if drift_metrics['is_anomaly'] else "🟢 Normal"
                    ]
                }
                
                st.dataframe(metrics_data, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                st.markdown("""
                ### Metrics Explanation
                
                - **Expected Sales**: ML model prediction based on current conditions
                - **Drift Score**: Measures how much current sales deviate from historical patterns (higher = more drift)
                - **Z-Score**: Number of standard deviations from mean (>2 or <-2 indicates anomaly)
                - **PSI (Population Stability Index)**: Measure of distribution shift (>0.2 = significant drift)
                - **Percentile**: Current sales position relative to historical data (50% = median)
                - **Temperature Drift**: How much current temperature differs from historical average
                """)

if __name__ == "__main__":
    main()
