import os
from datetime import datetime

class ModelRegistry:
    def __init__(self, base_path: str = "models/"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model, model_type: str, version: str = None):
        """Mock model saving"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Model saved: {model_type} version {version}")
        return f"{self.base_path}/{model_type}/{version}"
    
    def load_latest_model(self, model_type: str):
        """Mock model loading"""
        print(f"Loading latest {model_type} model")
        return None