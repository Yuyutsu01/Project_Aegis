"""
FYERS API Configuration - Now imports from api_keys.py
"""
from .api_keys import FYERS_CLIENT_ID, FYERS_SECRET_KEY, FYERS_REDIRECT_URI

FYERS_CREDENTIALS = {
    'client_id': FYERS_CLIENT_ID,
    'secret_key': FYERS_SECRET_KEY,
    'redirect_uri': FYERS_REDIRECT_URI
}