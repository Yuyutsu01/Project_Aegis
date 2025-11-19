import sqlite3
import pandas as pd
from datetime import datetime
import os

class LocalDatabase:
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                action TEXT,
                quantity REAL,
                price REAL,
                reason TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, symbol: str, action: str, quantity: float, price: float, reason: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, action, quantity, price, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), symbol, action, quantity, price, reason))
        
        conn.commit()
        conn.close()
    
    def get_trade_history(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
        conn.close()
        return df