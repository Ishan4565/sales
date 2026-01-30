import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sqlalchemy import create_engine
import os

# Use the environment variable set in Render/Hugging Face
db_url = os.getenv("postgresql://inventory_drift_db_user:Xf9BpwHH8zNTqmjap0W1bCLXKd3kUzni@dpg-d5uarfiqcgvc73asnf80-a.singapore-postgres.render.com/inventory_drift_db")

# Fallback for local testing if the environment variable isn't found
if db_url is None:
    db_url = 'postgresql://postgres:1234@localhost:5432/postgres'

engine = create_engine(db_url)

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
    
    model = GradientBoostingRegressor().fit(X, y)
    return model

def predict_and_log(product, season_name, temp, promo, past, actual_sales, model):
    mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    s_index = mapping[season_name]
    
    input_data = pd.DataFrame([[s_index, temp, promo, past]], 
                               columns=['season_index', 'temp', 'promo', 'past_sales'])
    
    prediction = model.predict(input_data)[0]
    drift = abs(prediction - actual_sales) / prediction
    status = "Critical Drift" if drift > 0.2 else "Stable"
    
    result = pd.DataFrame([{
        'product_name': product,
        'season': season_name,
        'predicted_demand': float(prediction),
        'actual_demand': float(actual_sales),
        'drift_score': float(drift),
        'status': status
    }])
    
    result.to_sql('inventory_monitor', engine, if_exists='append', index=False)
    return prediction, drift