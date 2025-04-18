# utils.py
from passlib.context import CryptContext

# Configure passlib context
# We'll use bcrypt for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

print("Password hashing utilities (verify_password, hash_password) defined.")
import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df['week'] = pd.to_datetime(df['week'])
    df['month'] = df['week'].dt.month
    df['day_of_week'] = df['week'].dt.dayofweek
    df['quarter'] = df['week'].dt.quarter
    df['discount'] = df['base_price'] - df['total_price']
    df['discount_percentage'] = (df['discount'] / df['base_price']).fillna(0)
    df['is_discounted'] = (df['discount'] > 0).astype(int)
    df['promo_display'] = df['is_featured_sku'] + df['is_display_sku']
    df['store_sku'] = df['store_id'].astype(str) + '_' + df['sku_id'].astype(str)
    df = df.drop('week', axis=1)
    return df

def encode_features(train_df, test_df, target_column):
    cat_features = ['store_id', 'sku_id', 'month', 'day_of_week', 'quarter']
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Target encoding for store_sku
    store_sku_mean = train_df.groupby('store_sku')[target_column].mean()
    train_df['store_sku_encoded'] = train_df['store_sku'].map(store_sku_mean)
    test_df['store_sku_encoded'] = test_df['store_sku'].map(store_sku_mean)
    global_mean = train_df[target_column].mean()
    test_df['store_sku_encoded'].fillna(global_mean, inplace=True)

    # One-hot encoding
    for col in cat_features:
        dummies = pd.get_dummies(pd.concat([train_df[col], test_df[col]]), prefix=col, drop_first=True)
        train_df = pd.concat([train_df, dummies.iloc[:len(train_df)]], axis=1)
        test_df = pd.concat([test_df, dummies.iloc[len(train_df):].reset_index(drop=True)], axis=1)

    # Drop original categorical columns
    drop_cols = cat_features + ['store_sku']
    train_df.drop(columns=drop_cols, inplace=True)
    test_df.drop(columns=drop_cols, inplace=True)

    return train_df, test_df
