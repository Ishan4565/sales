import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor

# Simple engine factory
def get_db_engine():
    # Use the variable you added to Render settings
    db_url = os.getenv("postgresql://inventory_drift_db_user:Xf9BpwHH8zNTqmjap0W1bCLXKd3kUzni@dpg-d5uarfiqcgvc73asnf80-a/inventory_drift_db")
    
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    
    # Only for local testing
    return create_engine('postgresql://postgres:1234@localhost:5432/postgres')

def train_universal_model():
    data = pd.DataFrame({
        'season_index': [1, 2, 3, 4, 1, 2, 3, 4],
        'temp': [15, 30, 10, -5, 18, 35, 8, -10],
        'promo': [0, 1, 0, 1, 1, 0, 0, 1],
        'past_sales': [300, 800, 400, 600, 350, 850, 420, 650],
        'actual': [320, 820, 390, 610, 360, 810, 400, 630]
    })
    X = data[['season_index', 'temp', 'promo', 'past_sales']]
    y = data['actual']
    return GradientBoostingRegressor().fit(X, y)

def predict_and_log(product, season_name, temp, promo, past, actual_sales, model):
    mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    s_index = mapping[season_name]
    
    input_data = pd.DataFrame([[s_index, temp, promo, past]], 
                               columns=['season_index', 'temp', 'promo', 'past_sales'])
    
    prediction = model.predict(input_data)[0]
    drift = abs(prediction - actual_sales) / (prediction + 1e-9)
    status = "Critical" if drift > 0.2 else "Stable"
    
    engine = get_db_engine()
    result = pd.DataFrame([{
        'product_name': product, 'season': season_name,
        'predicted_demand': float(prediction), 'actual_demand': float(actual_sales),
        'drift_score': float(drift), 'status': status
    }])
    
    result.to_sql('inventory_monitor', engine, if_exists='append', index=False)
    return prediction, drift
