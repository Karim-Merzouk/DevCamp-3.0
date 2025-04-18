# models.py
from sqlalchemy import Column, Integer, String, Boolean
from database import Base # Import Base from database.py

class User(Base):
    __tablename__ = "users" # Name of the table in the database

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    # Add other fields as needed, e.g., full_name, created_at, etc.

print("User model defined.")