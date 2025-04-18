# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List # Use standard 'list' in Python 3.9+
# Change this:
# from . import auth, database, models, schemas, utils

# To this:
import auth
import database
import models
import schemas
import utils
# Usage remains the same, e.g., database.get_db, auth.authenticate_user
from fastapi.middleware.cors import CORSMiddleware # Ensure this is imported
# from dotenv import load_dotenv # <--- Import load_dotenv
from dotenv import load_dotenv # <--- Import load_dotenv


# Load environment variables from .env file BEFORE other imports that might need them
load_dotenv() # <--- Add this line near the top
import pandas as pd

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
    df.drop('week', axis=1, inplace=True)
    return df

def encode_features(train_df, new_df, target_column):
    new_df = new_df.copy()
    train_df = train_df.copy()

    cat_features = ['store_id', 'sku_id', 'month', 'day_of_week', 'quarter', 'store_sku']

    for col in cat_features:
        if col == 'store_sku':
            means = train_df.groupby(col)[target_column].mean()
            global_mean = train_df[target_column].mean()
            new_df[f'{col}_encoded'] = new_df[col].map(means).fillna(global_mean)
        else:
            combined = pd.concat([train_df[col], new_df[col]], axis=0)
            dummies = pd.get_dummies(combined, prefix=col, drop_first=True)
            train_dummies = dummies.iloc[:len(train_df)]
            new_dummies = dummies.iloc[len(train_df):]
            new_dummies.reset_index(drop=True, inplace=True)
            new_df = pd.concat([new_df, new_dummies], axis=1)

    return train_df, new_df

# --- Now continue with your regular imports ---
app = FastAPI()

# --- CORS Configuration --- <--- MAKE SURE THIS IS PRESENT
# main.py
# ...
origins = [
    "http://localhost:5173", # Keep for local dev
    "http://127.0.0.1:5173", # Keep for local dev
    # Add your deployed frontend URL here when you know it
    # e.g., "https://your-frontend-app.onrender.com",
    # Add your Render backend URL here after deployment (optional but good practice)
    # e.g., "https://your-backend-api.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Update this list as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ...

# --- Database Initialization ---
# Create database tables if they don't exist
# In a production scenario with migrations (like Alembic), you might not do this here.

try:
    models.Base.metadata.create_all(bind=database.engine)
    print("Database tables checked/created successfully.")
except Exception as e:
    print(f"Error creating database tables: {e}")


# --- FastAPI App Instance ---
app = FastAPI(
    title="My Backend Project API",
    description="API for user registration and authentication.",
    version="0.1.0",
)

print("FastAPI app instance created.")


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """A simple root endpoint."""
    return {"message": "Welcome to the Backend Project API!"}

@app.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED, tags=["Users"])
async def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    """
    Register a new user (Inscription).
    """
    print(f"Attempting to register user: {user.username}")
    # Check if user already exists
    db_user_by_email = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user_by_email:
        print(f"Registration failed: Email '{user.email}' already registered.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    db_user_by_username = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user_by_username:
        print(f"Registration failed: Username '{user.username}' already registered.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")

    # Hash the password before saving
    hashed_password = utils.hash_password(user.password)
    # Create the new user instance (without plain password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        is_active=True # Or based on your logic (e.g., email verification needed)
    )
    # Add user to the session and commit to the database
    db.add(db_user)
    db.commit()
    db.refresh(db_user) # Refresh to get the ID generated by the DB
    print(f"User '{user.username}' registered successfully with ID: {db_user.id}")
    return db_user # Pydantic will automatically convert this based on response_model

@app.post("/token", response_model=schemas.Token, tags=["Authentication"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), # Inject form data (username/password)
    db: Session = Depends(database.get_db)
):
    """
    Log in a user (Connection) and return an access token.
    Client should send 'username' and 'password' as form data (not JSON).
    """
    print(f"Login attempt for user: {form_data.username}")
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}, # Standard header for 401
        )
    # Create JWT token
    access_token = auth.create_access_token(
        data={"sub": user.username} # 'sub' (subject) is standard JWT claim for user identifier
        # You can add more data to the token payload if needed, e.g., user roles
    )
    print(f"Login successful for '{form_data.username}', token generated.")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=schemas.User, tags=["Users"])
async def read_users_me(current_user: models.User = Depends(auth.get_current_active_user)):
    """
    Get the details of the currently authenticated user.
    Requires a valid Bearer token in the Authorization header.
    """
    print(f"Fetching profile for authenticated user: {current_user.username}")
    # The 'current_user' is already validated and retrieved by the dependency
    return current_user

# Example of another endpoint (optional)
@app.get("/users/", response_model=List[schemas.User], tags=["Users"])
async def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_active_user)):
    """
    Retrieve a list of users (example of pagination and protected endpoint).
    Requires authentication. In a real app, you might restrict this to admins.
    """
    print(f"User '{current_user.username}' requesting user list (skip={skip}, limit={limit}).")
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from model_utils import predict_units_sold

app = FastAPI()

class PredictionInput(BaseModel):
    store_id: int
    sku_id: int
    week: str  # Expecting "YYYY-MM-DD"
    base_price: float
    total_price: float
    is_featured_sku: int
    is_display_sku: int

@app.get("/")
def root():
    return {"message": "Sales Prediction API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        prediction = predict_units_sold(input_data.dict())
        return {"predicted_units_sold": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
print("API endpoints defined.")