import os
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="AEGIS | Cyber-Quant Terminal API")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DATA_PATH = os.path.join(BASE_DIR, "artifacts", "backtests", "meta_policy_results.csv")

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Aegis Terminal Offline. index.html not found."}
    return FileResponse(index_path)

@app.get("/api/data")
async def get_data():
    """Serves the backtest results as JSON."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Backtest data not found. Please run 'python run_meta_backtest.py' first.")
    
    try:
        df = pd.read_csv(DATA_PATH)
        # Convert date to string if it's not already
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        
        # Return as list of dicts for easy frontend consumption
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """Returns aggregated performance metrics."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Data not found.")
    
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Calculate overall metrics
        final_equity = df["equity"].iloc[-1]
        initial_equity = df["equity"].iloc[0]
        total_return = (final_equity / initial_equity) - 1
        
        # Sharpe (simplified)
        returns = df["pnl"]
        sharpe = (returns.mean() / returns.std() * (252**0.5)) if returns.std() != 0 else 0
        
        # Max Drawdown
        max_dd = df["drawdown"].min()
        
        # Win Rate
        win_rate = (df["pnl"] > 0).mean()
        
        return {
            "total_return": round(total_return * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "final_equity": round(final_equity, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
