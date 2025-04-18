import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Load all components
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = "xgb_model.pkl"

def load_pickle(filename):
    with open(os.path.join(filename), "rb") as f:
        return pickle.load(f)

model = load_pickle("xgb_model.pkl")
imputer = load_pickle("imputer.pkl")
scaler = load_pickle("scaler.pkl")
model_columns = load_pickle("X_columns.pkl")
cat_encoding_maps = load_pickle("cat_encoding_maps.pkl")

def preprocess_input(input_data: dict):
    df = pd.DataFrame([input_data])

    # Date processing
    if 'week' in df.columns:
        df['week'] = pd.to_datetime(df['week'])
        df['month'] = df['week'].dt.month
        df['day_of_week'] = df['week'].dt.dayofweek
        df['quarter'] = df['week'].dt.quarter
        df.drop('week', axis=1, inplace=True)

    # Pricing
    df['discount'] = df['base_price'] - df['total_price']
    df['discount_percentage'] = (df['discount'] / df['base_price']).fillna(0)
    df['is_discounted'] = (df['discount'] > 0).astype(int)

    # Promotions
    df['promo_display'] = df['is_featured_sku'] + df['is_display_sku']

    # Store-SKU Interaction
    df['store_sku'] = df['store_id'].astype(str) + '_' + df['sku_id'].astype(str)

    # Encodings
    for col in ['store_id', 'sku_id', 'month', 'day_of_week', 'quarter', 'store_sku']:
        if col in df.columns and col in cat_encoding_maps:
            if col == 'store_sku':
                mapping = cat_encoding_maps[col]['mapping']
                global_mean = cat_encoding_maps[col]['global_mean']
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(global_mean)
            else:
                dummy_cols = cat_encoding_maps[col].get('dummy_columns', [])
                for dummy_col in dummy_cols:
                    prefix = f"{col}_"
                    if dummy_col.startswith(prefix):
                        suffix = dummy_col[len(prefix):]
                        val = int(suffix) if suffix.isdigit() else suffix
                        df[dummy_col] = (df[col] == val).astype(int)

    df.drop(columns=['store_id', 'sku_id', 'month', 'day_of_week', 'quarter', 'store_sku'], errors='ignore', inplace=True)

    model_df = pd.DataFrame(0, index=df.index, columns=model_columns)
    for col in df.columns:
        if col in model_columns:
            model_df[col] = df[col]

    X_imputed = imputer.transform(model_df)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled

def predict_units_sold(input_data: dict) -> float:
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]
    return max(0, prediction)
