# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_USERNAME=os.getenv('DB_USERNAME')
    DB_PASSWORD=os.getenv('DB_PASSWORD')
    DB_HOST=os.getenv('DB_HOST')
    DB_NAME=os.getenv('DB_NAME')
    DB_PORT=os.getenv('DB_PORT')
    # SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    # SQLALCHEMY_TRACK_MODIFICATIONS = False

    # SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require'
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')
    REFRESH_TOKEN_SECRET = os.getenv('REFRESH_TOKEN_SECRET')