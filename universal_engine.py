import os

import pandas as pd

from sqlalchemy import create_engine



def get_db_engine():

    # Force the app to look for the variable AGAIN right now

    db_url = os.getenv("postgresql://inventory_drift_db_user:Xf9BpwHH8zNTqmjap0W1bCLXKd3kUzni@dpg-d5uarfiqcgvc73asnf80-a/inventory_drift_db")

    

    if db_url:

        # Essential Render Fix

        if db_url.startswith("postgres://"):

            db_url = db_url.replace("postgres://", "postgresql://", 1)

        return create_engine(db_url)

    else:

        # This is what's happening now; it's defaulting to localhost

        return create_engine('postgresql://postgres:1234@localhost:5432/postgres')



def predict_and_log(product, season_name, temp, promo, past, actual_sales, model):

    # ... (Keep your mapping and prediction logic the same) ...

    

    # NEW: Create engine ONLY when this function is called

    engine = get_db_engine()

    

    result = pd.DataFrame([{

        'product_name': product,

        'season': season_name,

        'predicted_demand': float(prediction),

        'actual_demand': float(actual_sales),

        'drift_score': float(drift),

        'status': status

    }])

    

    # This will now use the Cloud URL if it exists

    result.to_sql('inventory_monitor', engine, if_exists='append', index=False)

    return prediction, drift
