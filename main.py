from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# === FastAPI app ===
app = FastAPI(title="Sales Prediction API")

# === Required model files ===
REQUIRED_FILES = ['xgb_model.pkl', 'imputer.pkl', 'scaler.pkl', 'X_columns.pkl', 'cat_encoding_maps.pkl']

for file in REQUIRED_FILES:
    if not os.path.exists(file):
        raise RuntimeError(f"Missing required file: {file}. Please generate it using the create-pkl-files script.")

# === Load components ===
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('X_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)
with open('cat_encoding_maps.pkl', 'rb') as f:
    cat_encoding_maps = pickle.load(f)

# === Pydantic Schema ===
class PredictionInput(BaseModel):
    store_id: int
    sku_id: int
    week: str  # format: YYYY-MM-DD
    base_price: float
    total_price: float
    is_featured_sku: int
    is_display_sku: int

# === Preprocessing Function ===
def preprocess_input(input_data):
    df = pd.DataFrame([input_data.dict()])
    
    df['week'] = pd.to_datetime(df['week'])
    df['month'] = df['week'].dt.month
    df['day_of_week'] = df['week'].dt.dayofweek
    df['quarter'] = df['week'].dt.quarter
    df.drop('week', axis=1, inplace=True)

    df['discount'] = df['base_price'] - df['total_price']
    df['discount_percentage'] = (df['discount'] / df['base_price']).fillna(0)
    df['is_discounted'] = (df['discount'] > 0).astype(int)
    df['promo_display'] = df['is_featured_sku'] + df['is_display_sku']
    df['store_sku'] = df['store_id'].astype(str) + '_' + df['sku_id'].astype(str)

    for col in ['store_id', 'sku_id', 'month', 'day_of_week', 'quarter', 'store_sku']:
        if col in df.columns and col in cat_encoding_maps:
            if col == 'store_sku':
                mapping = cat_encoding_maps[col]['mapping']
                global_mean = cat_encoding_maps[col]['global_mean']
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(global_mean)
            else:
                dummy_cols = cat_encoding_maps[col].get('dummy_columns', [])
                for dummy_col in dummy_cols:
                    suffix = dummy_col[len(f"{col}_"):]
                    val = int(suffix) if suffix.isdigit() else suffix
                    df[dummy_col] = (df[col] == val).astype(int)

    df.drop(columns=['store_id', 'sku_id', 'month', 'day_of_week', 'quarter', 'store_sku'], errors='ignore', inplace=True)

    model_df = pd.DataFrame(0, index=df.index, columns=model_columns)
    for col in df.columns:
        if col in model_columns:
            model_df[col] = df[col]

    X_imputed = imputer.transform(model_df)
    X_processed = scaler.transform(X_imputed)

    return X_processed

# === Prediction Endpoint ===
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        print("🟡 Preprocessing input")
        processed = preprocess_input(input_data)
        print("🟢 Preprocessing done")

        print("🟡 Predicting")
        prediction = model.predict(processed)
        final = float(max(0, prediction[0]))  # Ensure native float for FastAPI
        print("🟢 Prediction complete")

        return {"predicted_units_sold": final}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
