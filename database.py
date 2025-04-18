# # database.py
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# import os

# # --- Database Configuration ---
# # Use SQLite for simplicity. For production, use PostgreSQL or MySQL.
# # Example for PostgreSQL: SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"
# # Example for MySQL: SQLALCHEMY_DATABASE_URL = "mysql+mysqlclient://user:password@mysqlserver/db"

# # DATABASE_URL = "postgresql://user:password@postgresserver/db" # Example for PostgreSQL
# DATABASE_URL = "sqlite:///./sql_app.db" # SQLite database file will be created in the project root

# # --- SQLAlchemy Engine Setup ---
# # `connect_args` is needed only for SQLite to allow usage across threads.
# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False} # Remove this line if not using SQLite
# )

# # --- Session Factory ---
# # Each instance of SessionLocal will be a database session.
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # --- Base Class for Models ---
# # Our ORM models will inherit from this class.
# Base = declarative_base()

# # --- Dependency for FastAPI Routes ---
# # Function to get a DB session for a request.
# # Ensures the session is closed after the request is finished.
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# print(f"Database URL: {DATABASE_URL}")
# print("Database engine and session configured.")
# database.py
import os # <--- Import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Database Configuration ---
# Get the database URL from the environment variable 'DATABASE_URL'
# Provide a default fallback (e.g., local SQLite) ONLY for local testing if needed
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db") # <--- CHANGE THIS

print(f"Attempting to connect to: {DATABASE_URL}") # Add for debugging

# Check if using PostgreSQL and adjust engine creation if needed (usually not necessary)
if DATABASE_URL.startswith("postgresql"):
     engine = create_engine(DATABASE_URL)
     print("Configured SQLAlchemy engine for PostgreSQL.")
elif DATABASE_URL.startswith("sqlite"):
     # Keep the connect_args for SQLite if it's the fallback
     engine = create_engine(
         DATABASE_URL, connect_args={"check_same_thread": False}
     )
     print("Configured SQLAlchemy engine for SQLite.")
else:
     # Default or handle other dialects if necessary
     engine = create_engine(DATABASE_URL)
     print("Configured SQLAlchemy engine for provided URL.")


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()