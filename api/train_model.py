import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Ensure local mlruns directory exists if using local tracking URI
if MLFLOW_TRACKING_URI.startswith("http://localhost") or MLFLOW_TRACKING_URI.startswith("file:") or MLFLOW_TRACKING_URI == "mlruns":
    mlruns_path = os.path.join(os.getcwd(), "mlruns")
    if not os.path.exists(mlruns_path):
        os.makedirs(mlruns_path)

def extract_features(df):
    # Select and encode relevant features
    features = [
        "amount",
        "merchant_category_code",
        "transaction_type"
    ]
    df = df.copy()
    for col in ["merchant_category_code", "transaction_type"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df[features]

def train_model():
    """Train an anomaly detection model using Isolation Forest."""
    print("🔍 Training anomaly detection model...")

    # Load producer-generated data
    data_path = os.getenv("PRODUCER_DATA_PATH", "../../producer/producer_transactions.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Producer data not found at {data_path}")
    df = pd.read_csv(data_path)
    X = extract_features(df)

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
    pipeline.fit(X)

    # Log the model with MLflow
    with mlflow.start_run(run_name="anomaly_detection_pipeline") as run:
        mlflow.log_param("n_estimators", pipeline.named_steps['model'].n_estimators)
        mlflow.log_param("contamination", pipeline.named_steps['model'].contamination)

        predictions = pipeline.predict(X)
        anomaly_ratio = (predictions == -1).sum() / len(predictions)
        print(f"Anomaly ratio on training data: {anomaly_ratio:.4f}")
        mlflow.log_metric("anomaly_ratio", anomaly_ratio)
        mlflow.log_metric("n_samples_trained", len(X))

        # Explicitly set artifact_path and print artifact URI for debugging
        artifact_path = "model"
        print(f"Logging model to artifact_path: {artifact_path}")
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path,
            registered_model_name="anomaly_detector"
        )
        print(f"Artifacts should be stored in: {mlflow.get_artifact_uri(artifact_path)}")

        print(f"✅ Pipeline trained and logged in MLflow. Run ID: {run.info.run_id}")
        print(f"✅ Pipeline registered as 'anomaly_detector'")

if __name__ == "__main__":
    train_model()