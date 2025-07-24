import os
from sqlmodel import SQLModel, create_engine
from app import DATABASE_FILE_PATH, Patient, Report # Import necessary components from app.py

# Ensure the directory for the database file exists
os.makedirs(os.path.dirname(DATABASE_FILE_PATH), exist_ok=True)

# Create the engine using the same path as in app.py
engine = create_engine(f"sqlite:///{DATABASE_FILE_PATH}")

def create_db_and_tables():
    """Creates all database tables defined by SQLModel."""
    print(f"Attempting to create database tables at {DATABASE_FILE_PATH}...")
    SQLModel.metadata.create_all(engine)
    print("✅ SQLite database tables created/checked!")

if __name__ == "__main__":
    create_db_and_tables()
