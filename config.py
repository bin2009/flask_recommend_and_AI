# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:290321@localhost/pbl6'  # Thay đổi thông tin kết nối
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')
    REFRESH_TOKEN_SECRET = os.getenv('REFRESH_TOKEN_SECRET')