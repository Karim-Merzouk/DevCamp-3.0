# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- User Schemas ---
# Base properties shared by other User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr # Pydantic validates email format

# Properties required during user creation (received via API)
class UserCreate(UserBase):
    password: str # Plain password received from client

# Properties to return via API (never include password hash)
class User(UserBase):
    id: int
    is_active: bool

    # Pydantic V2: Use model_config instead of Config
    # model_config = {
    #     "from_attributes": True # Allows Pydantic to read data from ORM models (obj.id instead of obj['id'])
    # }

    # Pydantic V1: Use Config
    class Config:
         orm_mode = True # Allows Pydantic to read data from ORM models (obj.id instead of obj['id'])

print("Pydantic schemas (Token, TokenData, UserBase, UserCreate, User) defined.")