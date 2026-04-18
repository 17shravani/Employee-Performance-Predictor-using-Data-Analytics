import os
import subprocess
import time
import sys
import threading

def run_backend():
    print("[*] Starting FastAPI Backend Server on port 8000...")
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    print("[*] Starting Streamlit Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])

if __name__ == "__main__":
    print("="*50)
    print("[*] INITIALIZING OPTIMA-HR ECOSYSTEM")
    print("="*50)
    
    # 1. Generate Data
    print("\n[Step 1] Initializing Synthetic Data Simulation...")
    from src.data.generator import generate_hr_data
    generate_hr_data()
    
    # 2. Train Model
    print("\n[Step 2] Executing Machine Learning Pipeline...")
    from src.ml.pipeline import train_and_save_model
    train_and_save_model()
    
    # 3. Start Backend
    print("\n[Step 3] Booting API and AI Agents...")
    api_thread = threading.Thread(target=run_backend, daemon=True)
    api_thread.start()
    
    # Give the backend a couple of seconds to spin up
    time.sleep(3)
    
    # 4. Start Frontend
    print("\n[Step 4] Launching Executive Dashboard...")
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n shutting down ecosystem...")
        sys.exit(0)
