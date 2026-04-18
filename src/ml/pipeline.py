import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_save_model():
    print("[*] Starting Model Training Pipeline...")
    
    # 1. Load Data
    data_path = 'data/synthetic_hr_data.csv'
    if not os.path.exists(data_path):
        print(f"[!] Error: Dataset not found at {data_path}. Run generator.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Separate Features and Target
    # Drop employee_id as it's an identifier
    X = df.drop(columns=['employee_id', 'target_perf_band'])
    y = df['target_perf_band']
    
    # 3. Identify Column Types
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['number']).columns
    
    # 4. Construct Preprocessing Pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # 5. Full Pipeline with Random Forest Classifier
    # 'balanced' class_weight is critical for HR data where 'Medium' dominates
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    # 6. Hyperparameter Tuning using Stratified K-Fold
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [8, 12, None],
        'clf__min_samples_leaf': [1, 3]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Fit Model
    print("[*] Tuning model hyperparameters (this may take a moment)...")
    gs.fit(X_train, y_train)
    
    best_model = gs.best_estimator_
    print(f"[*] Best CV F1-Macro Score: {gs.best_score_:.3f}")
    
    # 7. Evaluate on Test Set
    y_pred = best_model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # 8. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/employee_perf_model.pkl')
    print("[*] Model saved to 'models/employee_perf_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()
