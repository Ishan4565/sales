import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

print("="*80)
print("SALES FORECASTING SYSTEM")
print("="*80)

print("\nSTEP 1: Connecting to PostgreSQL")
print("-"*80)

password = 'ishan123'
encoded_password = quote_plus(password)
engine = create_engine(f'postgresql+psycopg2://postgres:{encoded_password}@localhost:5432/postgres')

print("✓ Connected to database")

print("\nSTEP 2: Loading Data from SQL")
print("-"*80)

query = """
SELECT 
    s.sale_id,
    s.sale_date,
    s.quantity,
    s.total_amount,
    s.discount_percent,
    p.product_name,
    p.category,
    p.unit_price,
    c.customer_type,
    c.region
FROM sales s
JOIN products p ON s.product_id = p.product_id
JOIN customers c ON s.customer_id = c.customer_id
ORDER BY s.sale_date
"""

df = pd.read_sql(query, engine)

print(f"✓ Loaded {len(df)} sales records")
print(f"Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
print("\nFirst 5 rows:")
print(df.head())

print("\nSTEP 3: Feature Engineering")
print("-"*80)

df['sale_date'] = pd.to_datetime(df['sale_date'])

df['year'] = df['sale_date'].dt.year
df['month'] = df['sale_date'].dt.month
df['day'] = df['sale_date'].dt.day
df['day_of_week'] = df['sale_date'].dt.dayofweek
df['quarter'] = df['sale_date'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['day_of_year'] = df['sale_date'].dt.dayofyear

df['revenue_per_unit'] = df['total_amount'] / df['quantity']
df['discount_amount'] = df['total_amount'] * (df['discount_percent'] / 100)

category_dummies = pd.get_dummies(df['category'], prefix='cat')
region_dummies = pd.get_dummies(df['region'], prefix='region')
customer_type_dummies = pd.get_dummies(df['customer_type'], prefix='cust')

df = pd.concat([df, category_dummies, region_dummies, customer_type_dummies], axis=1)

print(f"✓ Created {len(df.columns)} features")
print("\nKey features:")
print("  - Time features: year, month, day, quarter, day_of_week")
print("  - Derived features: revenue_per_unit, discount_amount")
print("  - Categorical: category, region, customer_type (one-hot encoded)")

print("\nSTEP 4: Preparing Data for ML")
print("-"*80)

feature_cols = ['year', 'month', 'day', 'day_of_week', 'quarter', 
                'is_weekend', 'day_of_year', 'quantity', 'unit_price',
                'discount_percent', 'revenue_per_unit']

feature_cols += [col for col in df.columns if col.startswith(('cat_', 'region_', 'cust_'))]

X = df[feature_cols]
y = df['total_amount']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")

print("\nSTEP 5: Training Multiple Models")
print("-"*80)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }
    
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R² Score: {r2:.4f}")

print("\nSTEP 6: Model Comparison")
print("-"*80)

best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_metrics = results[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {best_metrics['r2']:.4f}")
print(f"   MAE: ${best_metrics['mae']:.2f}")
print(f"   RMSE: ${best_metrics['rmse']:.2f}")

print("\nAll Models Comparison:")
for name, metrics in results.items():
    print(f"  {name}: R²={metrics['r2']:.4f}, MAE=${metrics['mae']:.2f}")

print("\nSTEP 7: Feature Importance")
print("-"*80)

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

print("\nSTEP 8: Forecasting Future Sales")
print("-"*80)

last_date = df['sale_date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

future_data = []
for date in future_dates:
    avg_quantity = df['quantity'].mean()
    avg_unit_price = df['unit_price'].mean()
    avg_discount = df['discount_percent'].mean()
    
    sample_row = {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'day_of_week': date.dayofweek,
        'quarter': (date.month - 1) // 3 + 1,
        'is_weekend': 1 if date.dayofweek >= 5 else 0,
        'day_of_year': date.timetuple().tm_yday,
        'quantity': avg_quantity,
        'unit_price': avg_unit_price,
        'discount_percent': avg_discount,
        'revenue_per_unit': avg_unit_price
    }
    
    for col in feature_cols:
        if col.startswith(('cat_', 'region_', 'cust_')):
            sample_row[col] = 0
    
    if 'cat_Electronics' in feature_cols:
        sample_row['cat_Electronics'] = 1
    if 'region_North' in feature_cols:
        sample_row['region_North'] = 1
    if 'cust_Business' in feature_cols:
        sample_row['cust_Business'] = 1
    
    future_data.append(sample_row)

future_df = pd.DataFrame(future_data)
future_df = future_df[feature_cols]

if best_model_name == 'Linear Regression':
    future_scaled = scaler.transform(future_df)
    future_predictions = best_model.predict(future_scaled)
else:
    future_predictions = best_model.predict(future_df)

print(f"✓ Generated forecast for next 30 days")
print(f"\nForecast Summary:")
print(f"  Total predicted sales: ${sum(future_predictions):,.2f}")
print(f"  Average daily sales: ${np.mean(future_predictions):,.2f}")
print(f"  Min predicted: ${min(future_predictions):,.2f}")
print(f"  Max predicted: ${max(future_predictions):,.2f}")

print("\nNext 7 days forecast:")
for i in range(7):
    print(f"  {future_dates[i].strftime('%Y-%m-%d')}: ${future_predictions[i]:,.2f}")

print("\nSTEP 9: Saving Results to Database")
print("-"*80)

forecast_df = pd.DataFrame({
    'forecast_date': future_dates,
    'predicted_sales': future_predictions,
    'model_used': best_model_name,
    'created_at': datetime.now()
})

forecast_df.to_sql('sales_forecast', engine, if_exists='replace', index=False)
print("✓ Forecast saved to table: sales_forecast")

historical_predictions = best_model.predict(X_test_scaled if best_model_name == 'Linear Regression' else X_test)

test_results = pd.DataFrame({
    'actual_sales': y_test.values,
    'predicted_sales': historical_predictions,
    'difference': y_test.values - historical_predictions,
    'percentage_error': ((y_test.values - historical_predictions) / y_test.values * 100)
})

test_results.to_sql('model_performance', engine, if_exists='replace', index=False)
print("✓ Performance metrics saved to table: model_performance")

print("\nSTEP 10: Saving Model Files")
print("-"*80)

joblib.dump(best_model, 'sales_forecast_model.pkl')
joblib.dump(scaler, 'sales_forecast_scaler.pkl')

metadata = {
    'model_name': best_model_name,
    'feature_columns': feature_cols,
    'metrics': {
        'mae': best_metrics['mae'],
        'rmse': best_metrics['rmse'],
        'r2': best_metrics['r2']
    },
    'training_date': datetime.now().strftime('%Y-%m-%d')
}

joblib.dump(metadata, 'sales_forecast_metadata.pkl')

print("✓ Saved: sales_forecast_model.pkl")
print("✓ Saved: sales_forecast_scaler.pkl")
print("✓ Saved: sales_forecast_metadata.pkl")

print("\n" + "="*80)
print("✅ SALES FORECASTING COMPLETE!")
print("="*80)

print(f"""
📊 Summary:
  • Historical data: {len(df)} sales records
  • Date range: {df['sale_date'].min()} to {df['sale_date'].max()}
  • Best model: {best_model_name}
  • Model accuracy: {best_metrics['r2']:.1%}
  • Average prediction error: ${best_metrics['mae']:.2f}
  • Forecast period: Next 30 days
  • Total forecasted sales: ${sum(future_predictions):,.2f}

🗄️ Database Tables Created:
  • sales (original sales data)
  • products (product catalog)
  • customers (customer data)
  • sales_forecast (30-day predictions)
  • model_performance (accuracy metrics)

📁 Files Saved:
  • sales_forecast_model.pkl
  • sales_forecast_scaler.pkl
  • sales_forecast_metadata.pkl

🎯 Next Steps:
  1. Query forecast: SELECT * FROM sales_forecast
  2. Check accuracy: SELECT * FROM model_performance
  3. Use model for new predictions
  4. Deploy with Streamlit (next step!)
""")

print("="*80)