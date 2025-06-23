import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline  # Added import

# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic transaction data for training the anomaly detection model."""
    # Normal transaction data
    normal_data = np.random.normal(loc=100, scale=50, size=(int(n_samples * 0.95), 6))
    
    # Anomalous transaction data
    anomalous_data = np.random.normal(loc=500, scale=200, size=(int(n_samples * 0.05), 6))
    
    # Combine data
    data = np.vstack([normal_data, anomalous_data])
    
    # Create DataFrame with feature names
    df = pd.DataFrame(
        data,
        columns=[
            'amount',
            'avg_amount',
            'relative_size',
            'max_amount',
            'transaction_count',
            'velocity'
        ]
    )
    
    return df

def train_model():
    """Train an anomaly detection model using Isolation Forest."""
    print("🔍 Training anomaly detection model...")
    
    # Generate synthetic data for training
    data = generate_synthetic_data()
    
    # Create a pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,  # Expected ratio of anomalies
            random_state=42
        ))
    ])
    
    # Fit the pipeline
    pipeline.fit(data)
    
    # Log the model with MLflow
    with mlflow.start_run(run_name="anomaly_detection_pipeline") as run:  # Updated run_name
        # Log model parameters from the model step in the pipeline
        mlflow.log_param("n_estimators", pipeline.named_steps['model'].n_estimators)
        mlflow.log_param("contamination", pipeline.named_steps['model'].contamination)
        
        # Log synthetic performance metrics
        # In a real scenario, you would evaluate on a test set
        predictions = pipeline.predict(data)
        anomaly_ratio = (predictions == -1).sum() / len(predictions)
        
        mlflow.log_metric("anomaly_ratio", anomaly_ratio)
        mlflow.log_metric("n_samples_trained", len(data))
        
        # Log the pipeline
        mlflow.sklearn.log_model(
            pipeline,  # Log the entire pipeline
            "model",  # Changed artifact_path to "model" to match loading path
            registered_model_name="anomaly_detector"
        )
        
        print(f"✅ Pipeline trained and logged in MLflow. Run ID: {run.info.run_id}")
        print(f"✅ Pipeline registered as 'anomaly_detector'")

if __name__ == "__main__":
    train_model()