# auth.py
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext # Already defined in utils, but import here for clarity if needed
from sqlalchemy.orm import Session
import auth
import database
import models
import schemas
import utils
# auth.py
import os # <--- Import os
# ... other imports ...

# --- Configuration ---
# Load secret key from environment variable. Provide a default ONLY for local dev if necessary,
# BUT STRONGLY recommend setting it locally too via .env or similar.
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_insecure_default_key_CHANGE_THIS") # <--- CHANGE THIS
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

if SECRET_KEY == "a_very_insecure_default_key_CHANGE_THIS":
    print("\n*** WARNING: Using default insecure SECRET_KEY! Set the SECRET_KEY environment variable. ***\n")



# OAuth2 Password Flow requires a tokenUrl
# This is the relative URL endpoint that the client will use to send username/password
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Matches the endpoint in main.py

# --- Helper Functions ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Default expiration time if none provided
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str) -> Optional[models.User]:
    """Fetches a user by username from the database."""
    return db.query(models.User).filter(models.User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[models.User]:
    """Authenticates a user by username and password."""
    user = get_user(db, username)
    if not user:
        print(f"Authentication failed: User '{username}' not found.")
        return None # User not found
    if not utils.verify_password(password, user.hashed_password):
        print(f"Authentication failed: Incorrect password for user '{username}'.")
        return None # Incorrect password
    print(f"Authentication successful for user '{username}'.")
    return user

# --- Dependency for Protected Routes ---
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    """
    Dependency function to get the current user from a JWT token.
    Used to protect endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub") # "sub" is standard claim for subject (username)
        if username is None:
            print("Token decoding error: Username ('sub') claim missing.")
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError as e:
         print(f"Token decoding error: {e}")
         raise credentials_exception

    # Get the user from the database using the username from the token
    user = get_user(db, username=token_data.username)
    if user is None:
        print(f"Token validation error: User '{token_data.username}' not found in DB.")
        raise credentials_exception
    print(f"Token validated successfully for user '{user.username}'.")
    return user

async def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    """
    Dependency that checks if the user obtained from the token is active.
    Builds upon get_current_user.
    """
    if not current_user.is_active:
        print(f"Authorization error: User '{current_user.username}' is inactive.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    print(f"Active user check passed for '{current_user.username}'.")
    return current_user


print("Authentication functions (create_access_token, authenticate_user, get_current_user, etc.) defined.")