from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="OptimaHR Performance Engine REST API",
    description="Inference layer serving ML predictions for Employee Performance.",
    version="1.0.0"
)

# Load the model outside the request loop for latency optimization
MODEL_PATH = "models/employee_perf_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[*] ML Model loaded into memory securely.")
    else:
        print(f"[!] Warning: Model not found at {MODEL_PATH}. Prediction endpoints will fail.")

# Schema for incoming employee data
class EmployeeData(BaseModel):
    department: str
    job_level: str
    experience_years: float
    on_time_delivery_rate: float
    bug_count: int
    training_hours: int
    avg_login_hours: float
    peer_score: float
    manager_score: float

@app.get("/")
def health_check():
    return {"status": "OptimaHR Inference Microservice is live!"}

@app.post("/predict")
def predict_performance(data: EmployeeData):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Convert Pydantic object to format expected by scikit-learn pipeline (DataFrame)
    input_df = pd.DataFrame([data.dict()])
    
    try:
        prediction = model.predict(input_df)[0]
        # In a real enterprise system, predict_proba would be used for risk thresholds:
        probabilities = model.predict_proba(input_df)[0]
        class_mapping = model.classes_
        
        prob_dict = {class_mapping[i]: round(float(probabilities[i]), 3) for i in range(len(class_mapping))}
        
        return {
            "predicted_band": str(prediction),
            "confidence_scores": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
