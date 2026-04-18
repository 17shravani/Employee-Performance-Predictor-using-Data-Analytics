import pandas as pd
import numpy as np
import os

def generate_hr_data(num_records=5000):
    """
    Simulates a realistic corporate HR dataset with statistical relationships.
    Avoids data leakage by ensuring target variable is a logical resultant of the features.
    """
    print(f"[*] Generating {num_records} synthetic HR records...")
    
    np.random.seed(42)
    
    # 1. Demographics & Context
    employee_ids = [f"EMP_{str(i).zfill(5)}" for i in range(1, num_records + 1)]
    departments = np.random.choice(['Engineering', 'Sales', 'Marketing', 'Customer Support', 'HR'], num_records, p=[0.4, 0.25, 0.15, 0.15, 0.05])
    job_levels = np.random.choice(['Junior', 'Mid', 'Senior', 'Lead'], num_records, p=[0.3, 0.4, 0.2, 0.1])
    experience_years = np.round(np.random.uniform(0.5, 20.0, num_records), 1)
    
    # 2. Work Signals & Quality
    # We correlate delivery rate slightly with experience
    on_time_delivery_rate = np.clip(np.random.normal(0.6 + (experience_years * 0.01), 0.15), 0.1, 1.0)
    bug_count = np.where(departments == 'Engineering', np.random.poisson(lam=10), np.random.poisson(lam=2))
    # Higher experience -> fewer bugs
    bug_count = np.clip(bug_count - (experience_years * 0.2).astype(int), 0, None)
    
    # 3. Engagement & Setup
    training_hours = np.random.poisson(lam=15, size=num_records)
    avg_login_hours = np.random.normal(8.2, 1.5, num_records)
    
    # 4. Feedback
    peer_score = np.clip(np.random.normal(3.5, 0.8, num_records), 1, 5)
    manager_score = np.clip(np.random.normal(3.5, 0.8, num_records), 1, 5)
    
    # --- Generate Target Variable (Performance Band) Logic ---
    # Performance is heavily reliant on delivery rate, peer/manager score, and bugs (inversely)
    latent_perf_score = (
        (on_time_delivery_rate * 3) +
        (peer_score * 0.5) + 
        (manager_score * 0.8) +
        (training_hours * 0.02) - 
        (bug_count * 0.05) + 
        np.random.normal(0, 0.5, num_records) # noise
    )
    
    # Percentiles for High/Medium/Low bands
    p75 = np.percentile(latent_perf_score, 75)
    p25 = np.percentile(latent_perf_score, 25)
    
    perf_bands = []
    for score in latent_perf_score:
        if score >= p75:
            perf_bands.append("High")
        elif score <= p25:
            perf_bands.append("Low")
        else:
            perf_bands.append("Medium")

    # Construct DataFrame
    df = pd.DataFrame({
        'employee_id': employee_ids,
        'department': departments,
        'job_level': job_levels,
        'experience_years': experience_years,
        'on_time_delivery_rate': np.round(on_time_delivery_rate, 2),
        'bug_count': bug_count,
        'training_hours': training_hours,
        'avg_login_hours': np.round(avg_login_hours, 1),
        'peer_score': np.round(peer_score, 1),
        'manager_score': np.round(manager_score, 1),
        'target_perf_band': perf_bands
    })
    
    # Save to disk
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_hr_data.csv', index=False)
    print("[*] Successfully generated 'data/synthetic_hr_data.csv'.")
    return df

if __name__ == "__main__":
    generate_hr_data()
