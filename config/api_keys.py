"""
API Keys Configuration - Now reads from environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Read from environment variables
FYERS_CLIENT_ID = os.getenv('FYERS_CLIENT_ID')
FYERS_SECRET_KEY = os.getenv('FYERS_SECRET_KEY') 
FYERS_REDIRECT_URI = os.getenv('FYERS_REDIRECT_URI')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def validate_credentials():
    """Validate that required credentials are present"""
    missing = []
    
    if not FYERS_CLIENT_ID or 'your_actual' in FYERS_CLIENT_ID:
        missing.append('FYERS_CLIENT_ID')
    
    if not FYERS_SECRET_KEY or 'your_actual' in FYERS_SECRET_KEY:
        missing.append('FYERS_SECRET_KEY')
    
    if missing:
        print(f"❌ Missing credentials: {', '.join(missing)}")
        print("   Please update your .env file")
        return False
    
    print("✅ API credentials loaded from .env")
    return True