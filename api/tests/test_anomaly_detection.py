import pytest
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
import os

# Add the parent directory to sys.path to import app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mocking the Redis and MLflow dependencies since they're external services
class MockRedis:
    def __init__(self):
        self.data = {}
        self.sorted_sets = {}
        
    def zadd(self, key, mapping):
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        self.sorted_sets[key].update(mapping)
        return len(mapping)
    
    def zrangebyscore(self, key, min_score, max_score):
        if key not in self.sorted_sets:
            return []
        # Simple mock that returns all values as we're not testing time filtering
        return list(self.sorted_sets[key].keys())
    
    def zremrangebyscore(self, key, min_score, max_score):
        # We're not testing removal logic in the unit tests
        return 0
    
    def zcard(self, key):
        if key not in self.sorted_sets:
            return 0
        return len(self.sorted_sets[key])
    
    def ping(self):
        return True


# Create a test fixture for the model and redis client
@pytest.fixture
def setup_mocks(monkeypatch):
    # Create a mock isolation forest model
    mock_model = IsolationForest(contamination=0.05, random_state=42)
    # Fit with dummy data of 6 features, as expected by extract_features
    mock_model.fit(np.random.rand(100, 6)) 
    
    # Create a mock redis client
    mock_redis = MockRedis()
    
    # Patch the imports in the app module
    import app
    monkeypatch.setattr(app, 'model', mock_model)
    monkeypatch.setattr(app, 'redis_client', mock_redis)
    
    return app, mock_model, mock_redis


def test_extract_features(setup_mocks):
    app, _, _ = setup_mocks
    
    # Test transaction
    transaction = {
        'transaction_id': '123',
        'user_id': 'user-456',
        'amount': 100.0,
        'timestamp': '2023-01-01T12:00:00',
        'location': 'New York'
    }
    
    # Extract features
    feature_vector, feature_dict = app.extract_features(transaction)
    
    # Verify the shape of the feature vector
    assert feature_vector.shape == (1, 6)
    
    # Verify the feature dictionary has the expected keys
    expected_keys = ['amount', 'avg_amount', 'relative_size', 'max_amount', 'transaction_count', 'velocity']
    assert all(key in feature_dict for key in expected_keys)
    
    # Verify the amount is correctly captured in the features
    assert feature_dict['amount'] == 100.0


def test_is_anomalous_normal_transaction(setup_mocks):
    app, mock_model, _ = setup_mocks
    
    # Set up the model to predict '1' (normal) for our test data
    def mock_predict(X):
        return np.array([1])  # 1 means normal
    
    def mock_score_samples(X):
        return np.array([0.5])  # Positive score means normal
    
    mock_model.predict = mock_predict
    mock_model.score_samples = mock_score_samples
    
    # Test normal transaction
    transaction = {
        'transaction_id': '123',
        'user_id': 'user-456',
        'amount': 100.0,
        'timestamp': '2023-01-01T12:00:00',
        'location': 'New York'
    }
    
    # Check if it's anomalous
    is_anomaly, features, score = app.is_anomalous(transaction)
    
    # Verify the transaction is classified as normal
    assert is_anomaly == False
    assert score == 0.5


def test_is_anomalous_fraudulent_transaction(setup_mocks):
    app, mock_model, _ = setup_mocks
    
    # Set up the model to predict '-1' (anomaly) for our test data
    def mock_predict(X):
        return np.array([-1])  # -1 means anomaly
    
    def mock_score_samples(X):
        return np.array([-0.5])  # Negative score means anomaly
    
    mock_model.predict = mock_predict
    mock_model.score_samples = mock_score_samples
    
    # Test fraudulent transaction
    transaction = {
        'transaction_id': '456',
        'user_id': 'user-789',
        'amount': 10000.0,  # Unusually large amount
        'timestamp': '2023-01-01T12:00:00',
        'location': 'New York'
    }
    
    # Check if it's anomalous
    is_anomaly, features, score = app.is_anomalous(transaction)
    
    # Verify the transaction is classified as anomalous
    assert is_anomaly == True
    assert score == -0.5


def test_health_check_endpoint(setup_mocks):
    app, _, _ = setup_mocks

    # Create a test client using FastAPI's test client
    from fastapi.testclient import TestClient
    client = TestClient(app)  # Pass the FastAPI app directly

    # Test the health check endpoint
    response = client.get("/health")

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert data["components"]["model"] == "available"
    assert data["components"]["redis"] == "connected"