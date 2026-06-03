import subprocess
import sys
import os

def run_command(command):
    print(f"\n>>> Running: {command} ...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error executing: {command}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    skip_pipeline = os.environ.get("SKIP_PIPELINE", "false").lower() in ("true", "1", "yes")
    results_path = "artifacts/backtests/meta_policy_results.csv"
    results_exist = os.path.exists(results_path)

    if not skip_pipeline or not results_exist:
        if not results_exist and skip_pipeline:
            print(">>> Backtest results not found. Running the full pipeline first...")
        
        # Run entire execution pipeline sequentially
        run_command("python run_ingestion.py")
        run_command("python run_feature_engineer.py")
        run_command("python run_xgb_training.py")
        run_command("python run_ppo_training.py")
        run_command("python run_meta_backtest.py")
    else:
        print(">>> Skipping pipeline training/backtesting steps as results already exist.")

    print("\n>>> Launching Cyber-Quant Terminal Dashboard...")
    # Exec into Streamlit to replace Python process and handle signals correctly
    cmd = ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
    os.execvp(cmd[0], cmd)
