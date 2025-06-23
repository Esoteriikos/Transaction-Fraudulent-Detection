import os
import json
import time
import threading
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from kafka import KafkaConsumer, KafkaProducer
import redis
import mlflow
import mlflow.sklearn
import numpy as np
from prometheus_client import Counter, Histogram, start_http_server

# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="Real-time anomaly detection for financial transactions",
    version="1.0.0",
)

# Configuration from environment variables
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Configure Kafka
TRANSACTIONS_TOPIC = "transactions"
ALERTS_TOPIC = "alerts"

# Configure Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Prometheus metrics
TRANSACTIONS_COUNTER = Counter('transactions_processed_total', 'Total number of processed transactions')
FRAUDULENT_COUNTER = Counter('fraudulent_transactions_total', 'Total number of fraudulent transactions detected')
PROCESSING_TIME = Histogram('transaction_processing_seconds', 'Time spent processing transactions')

# Global model variable
model = None

def load_model():
    """Load the pre-trained anomaly detection model from MLflow."""
    global model
    try:
        # Use the latest model from MLflow
        model_uri = "models:/anomaly_detector/latest"
        print(f"Loading model from MLflow URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully")
        print(f"Model loaded from artifact location: {mlflow.get_artifact_uri(model_uri)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Do not use fallback model; fail fast so the issue is visible
        raise RuntimeError("Model could not be loaded from MLflow. Check artifact storage and configuration.") from e

def extract_features(transaction):
    """
    Extract features from a transaction and user history.
    
    Returns:
        tuple: (feature_vector, feature_dict)
    """
    user_id = transaction['user_id']
    amount = transaction['amount']
    
    # Get user's transaction history from Redis
    user_key = f"user:{user_id}:transactions"
    
    # Store the current transaction amount in Redis with timestamp as score
    current_timestamp = time.time()
    redis_client.zadd(user_key, {str(amount): current_timestamp})
    
    # Get recent transactions for this user (last minute)
    minute_ago = current_timestamp - 60
    recent_amounts = redis_client.zrangebyscore(user_key, minute_ago, '+inf')
    
    # Convert to float values
    recent_amounts = [float(amt) for amt in recent_amounts]
    
    # Feature calculations
    avg_amount = np.mean(recent_amounts) if recent_amounts else amount
    max_amount = np.max(recent_amounts) if recent_amounts else amount
    transaction_count = len(recent_amounts)
    
    # Calculate transaction velocity (transactions per minute)
    velocity_key = f"user:{user_id}:velocity"
    redis_client.zadd(velocity_key, {str(current_timestamp): current_timestamp})
    
    # Remove outdated entries
    redis_client.zremrangebyscore(velocity_key, 0, minute_ago)
    velocity = redis_client.zcard(velocity_key)
    
    # Prepare feature vector for model
    feature_vector = np.array([
        amount,
        avg_amount,
        amount / (avg_amount if avg_amount > 0 else 1),  # Relative size
        max_amount,
        transaction_count,
        velocity
    ]).reshape(1, -1)
    
    # Feature dictionary for logging
    feature_dict = {
        'amount': amount,
        'avg_amount': avg_amount,
        'relative_size': amount / (avg_amount if avg_amount > 0 else 1),
        'max_amount': max_amount,
        'transaction_count': transaction_count,
        'velocity': velocity
    }
    
    return feature_vector, feature_dict

def is_anomalous(transaction):
    """
    Determine if a transaction is anomalous based on features and model prediction.
    
    Returns:
        tuple: (is_anomaly (bool), features_dict, anomaly_score)
    """
    start_time = time.time()
    
    # Extract features
    feature_vector, feature_dict = extract_features(transaction)
    
    # Make prediction (-1 for anomaly, 1 for normal)
    prediction = model.predict(feature_vector)[0]
    anomaly_score = model.score_samples(feature_vector)[0]
    
    # Record processing time
    processing_time = time.time() - start_time
    PROCESSING_TIME.observe(processing_time)
    
    # Update Prometheus metrics
    TRANSACTIONS_COUNTER.inc()
    if prediction == -1:
        FRAUDULENT_COUNTER.inc()
    
    return prediction == -1, feature_dict, anomaly_score

def process_transaction_stream():
    """Background task to process the Kafka transaction stream."""
    # Configure Kafka consumer and producer
    consumer = KafkaConsumer(
        TRANSACTIONS_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='anomaly-detection-group',
        api_version=(0, 10)
    )
    
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        api_version=(0, 10)
    )
    
    print(f"🔍 Starting transaction processing from Kafka topic: {TRANSACTIONS_TOPIC}")
    
    for message in consumer:
        transaction = message.value
        
        # Process the transaction to detect if it's anomalous
        is_anomaly, features, anomaly_score = is_anomalous(transaction)
        
        # Prepare result for logging
        result = {
            'transaction_id': transaction['transaction_id'],
            'user_id': transaction['user_id'],
            'amount': transaction['amount'],
            'timestamp': transaction['timestamp'],
            'is_anomalous': bool(is_anomaly),  # Ensure native Python bool
            'anomaly_score': float(anomaly_score),
            'features': features,
            'detection_timestamp': datetime.now().isoformat()
        }
        
        # If anomalous, send an alert to the alerts topic
        if is_anomaly:
            try:
                producer.send(ALERTS_TOPIC, result)
                print(f"🚨 Alert sent for transaction {transaction['transaction_id']} (Score: {anomaly_score:.4f})")
            except Exception as e:
                print(f"❌ Failed to send alert: {e}")
        else:
            print(f"✅ Normal transaction {transaction['transaction_id']} (Score: {anomaly_score:.4f})")

@app.on_event("startup")
async def startup_event():
    """Initialize components when the application starts."""
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Load the anomaly detection model
    load_model()
    
    # Start transaction processing in a background thread
    threading.Thread(target=process_transaction_stream, daemon=True).start()

@app.get("/")
async def root():
    """Root endpoint that returns API status."""
    return {
        "status": "online",
        "service": "Anomaly Detection API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if model is loaded
    model_status = "available" if model is not None else "unavailable"
    
    # Check Redis connection
    try:
        redis_status = redis_client.ping()
    except:
        redis_status = False
    
    return {
        "status": "healthy",
        "components": {
            "model": model_status,
            "redis": "connected" if redis_status else "disconnected",
        },
        "metrics": {
            "transactions_processed": TRANSACTIONS_COUNTER._value.get(),
            "fraudulent_detected": FRAUDULENT_COUNTER._value.get(),
        }
    }

@app.post("/predict")
async def predict(transaction: dict):
    """Manual endpoint to test anomaly detection on a single transaction."""
    if not transaction.get('user_id') or not transaction.get('amount'):
        return {"error": "Invalid transaction data. Required fields: user_id, amount"}
    
    is_anomaly, features, score = is_anomalous(transaction)
    
    return {
        "transaction_id": transaction.get('transaction_id', 'manual'),
        "is_anomalous": is_anomaly,
        "anomaly_score": float(score),
        "features": features
    }