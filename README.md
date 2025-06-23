# Real-Time Anomaly Detection System

A complete end-to-end streaming data system that uses machine learning to detect anomalies in financial transactions in real-time.

![Architecture Diagram](https://via.placeholder.com/800x400?text=Real-Time+Anomaly+Detection+Architecture)
*(A more detailed architecture diagram would ideally be placed here, showing data flow between components)*

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
  - [Data Flow](#data-flow)
  - [Component Breakdown](#component-breakdown)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Deployment](#installation--deployment)
  - [Verifying the System](#verifying-the-system)
- [Training the Model](#training-the-model)
- [Development](#development)
  - [Project Structure](#project-structure)
  - [Key Files and Directories](#key-files-and-directories)
  - [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Monitoring](#monitoring)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- **Real-time data processing** with Kafka streams
- **Machine learning-based anomaly detection** using Isolation Forest algorithm
- **Stateful feature engineering** with Redis for fast access storage
- **Complete observability** with Prometheus and Grafana dashboards
- **MLOps practices** including model tracking with MLflow
- **CI/CD pipeline** with GitHub Actions for automated testing
- **Containerized deployment** with Docker and Docker Compose

## Technology Stack

- **Data Streaming**: Apache Kafka
- **In-memory Database**: Redis
- **API & Processing**: FastAPI, Python
- **Machine Learning**: scikit-learn (Isolation Forest)
- **Model Tracking**: MLflow
- **Metrics & Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## System Architecture

The system is designed as a microservices-based architecture, containerized using Docker and orchestrated with Docker Compose. It processes a continuous stream of financial transactions, applies machine learning for anomaly detection, and provides real-time monitoring and alerting capabilities.

### Data Flow

1.  **Transaction Generation**: The `producer` service simulates financial transactions. These transactions are JSON objects containing details like `transaction_id`, `user_id`, `amount`, `timestamp`, and `location`.
2.  **Kafka Ingestion**: Simulated transactions are published to the `transactions` Kafka topic. Kafka acts as a durable and scalable message broker, decoupling the producer from the consumer.
3.  **Real-time Processing (API Service)**:
    *   The `anomaly-detection-api` service consumes transactions from the `transactions` topic.
    *   For each transaction, it performs **feature engineering**. This involves:
        *   Retrieving historical transaction data for the user from **Redis** (e.g., average transaction amount, transaction frequency).
        *   Calculating new features based on the current transaction and historical context (e.g., transaction amount relative to average, velocity of transactions).
    *   The engineered features are then fed into a pre-trained **Isolation Forest** model (loaded via **MLflow**) to predict if the transaction is an anomaly.
    *   The prediction result (anomaly or normal), along with the original transaction data, features, and anomaly score, is prepared.
4.  **Alerting**: If a transaction is flagged as anomalous, an alert message is published to the `alerts` Kafka topic. This topic can be consumed by other services for further action (e.g., notifying security teams, blocking accounts).
5.  **Metrics Collection**: The `anomaly-detection-api` service exposes metrics (e.g., number of transactions processed, number of anomalies detected, processing latency) via a `/metrics` endpoint, which **Prometheus** scrapes.
6.  **Monitoring & Visualization**: **Grafana** queries Prometheus for these metrics and displays them on a real-time dashboard, providing insights into the system's health and performance.
7.  **Model Management**: **MLflow** is used to track experiments, log models during training (`train_model.py`), and serve the latest production model to the API service.

### Component Breakdown

*   **`producer`**: Python script using Faker to generate realistic transaction data and send it to Kafka.
*   **`kafka` & `zookeeper`**: Apache Kafka cluster for message queuing. Zookeeper is used for Kafka coordination.
*   **`redis`**: In-memory data store used for caching user transaction history for rapid feature calculation.
*   **`anomaly-detection-api`**: FastAPI application that:
    *   Consumes transactions from Kafka.
    *   Performs feature engineering using data from Redis.
    *   Uses an ML model (from MLflow) to detect anomalies.
    *   Publishes alerts to Kafka.
    *   Exposes Prometheus metrics.
*   **`mlflow`**: Platform for managing the ML lifecycle, including experiment tracking, model storage, and model serving.
*   **`prometheus`**: Time-series database for collecting metrics from the API service.
*   **`grafana`**: Visualization tool for creating dashboards based on Prometheus data.
*   **`docker-compose.yml`**: Defines and orchestrates all the services.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Installation & Deployment

1. Clone the repository:
   ```
   git clone <repository-url>
   cd anomaly-detection
   ```

2. Start the system using Docker Compose:
   ```bash
   docker-compose up -d
   ```
   This command will build the images (if not already built) and start all services in detached mode.

3. Access the different components:
   - **FastAPI Documentation (API Service)**: http://localhost:8000/docs
   - **MLflow UI**: http://localhost:5000
   - **Prometheus UI**: http://localhost:9090
   - **Grafana Dashboard**: http://localhost:3000 (username: `admin`, password: `admin`)

### Verifying the System

Once the services are up, you can verify their status:
*   **Check Docker Containers**: `docker-compose ps` should show all services as `Up` or `running`.
*   **Kafka Topics**: You might need to wait a few moments for Kafka to initialize. The `transactions` and `alerts` topics should be auto-created.
*   **Producer Logs**: `docker-compose logs producer` should show transactions being sent.
*   **API Logs**: `docker-compose logs anomaly-detection-api` should show messages about model loading, Kafka consumer starting, and processing transactions.
*   **Grafana Dashboard**: Open the Grafana dashboard. After a few minutes, you should see data populating the graphs (e.g., "Transaction Throughput").
*   **MLflow**: Ensure the `anomaly_detector` model is registered.

## Training the Model

The system uses a pre-trained Isolation Forest model. The training script (`api/train_model.py`) generates synthetic data and trains a new model, then logs it to MLflow and registers it.

To retrain or train an initial model:
```bash
docker-compose exec anomaly-detection-api python train_model.py
```
This command executes the `train_model.py` script inside the running `anomaly-detection-api` container. The API service will automatically load the latest registered version of the "anomaly_detector" model from MLflow upon startup or if it's designed to periodically refresh.

## Development

### Project Structure

```
anomaly-detection/
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI configuration for automated testing.
├── api/                      # Anomaly detection FastAPI service
│   ├── app.py                # Main FastAPI application: Kafka consumer, feature engineering, prediction logic, API endpoints.
│   ├── train_model.py        # Script for training the anomaly detection model and logging to MLflow.
│   ├── Dockerfile            # Docker build instructions for the API service.
│   ├── requirements.txt      # Python dependencies for the API service.
│   └── tests/                # Unit tests for the API service.
│       └── test_anomaly_detection.py # Pytest tests for feature extraction and anomaly detection logic.
├── producer/                 # Transaction data generator
│   ├── producer.py           # Python script to generate fake transactions and send to Kafka.
│   ├── Dockerfile            # Docker build instructions for the producer.
│   └── requirements.txt      # Python dependencies for the producer.
├── mlflow/                   # MLflow service configuration
│   └── Dockerfile            # Docker build instructions for the MLflow tracking server.
│   └── mlruns/               # (Typically gitignored) Local storage for MLflow experiment data and artifacts if not using remote storage.
├── prometheus/               # Prometheus configuration
│   └── prometheus.yml        # Prometheus scrape configurations (e.g., target for API metrics).
├── grafana/                  # Grafana provisioning and dashboards
│   └── provisioning/
│       ├── dashboards/
│       │   └── anomaly_dashboard.json # JSON definition of the main Grafana dashboard.
│       │   └── dashboards.yml         # Grafana dashboard provisioning configuration.
│       └── datasources/
│           └── datasource.yml         # Grafana data source (Prometheus) provisioning.
├── docker-compose.yml        # Docker Compose file to define and run all services.
└── README.md                 # This documentation file.
```

### Key Files and Directories

```
anomaly-detection/
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI configuration for automated testing.
├── api/                      # Anomaly detection FastAPI service
│   ├── app.py                # Main FastAPI application: Kafka consumer, feature engineering, prediction logic, API endpoints.
│   ├── train_model.py        # Script for training the anomaly detection model and logging to MLflow.
│   ├── Dockerfile            # Docker build instructions for the API service.
│   ├── requirements.txt      # Python dependencies for the API service.
│   └── tests/                # Unit tests for the API service.
│       └── test_anomaly_detection.py # Pytest tests for feature extraction and anomaly detection logic.
├── producer/                 # Transaction data generator
│   ├── producer.py           # Python script to generate fake transactions and send to Kafka.
│   ├── Dockerfile            # Docker build instructions for the producer.
│   └── requirements.txt      # Python dependencies for the producer.
├── mlflow/                   # MLflow service configuration
│   └── Dockerfile            # Docker build instructions for the MLflow tracking server.
│   └── mlruns/               # (Typically gitignored) Local storage for MLflow experiment data and artifacts if not using remote storage.
├── prometheus/               # Prometheus configuration
│   └── prometheus.yml        # Prometheus scrape configurations (e.g., target for API metrics).
├── grafana/                  # Grafana provisioning and dashboards
│   └── provisioning/
│       ├── dashboards/
│       │   └── anomaly_dashboard.json # JSON definition of the main Grafana dashboard.
│       │   └── dashboards.yml         # Grafana dashboard provisioning configuration.
│       └── datasources/
│           └── datasource.yml         # Grafana data source (Prometheus) provisioning.
├── docker-compose.yml        # Docker Compose file to define and run all services.
└── README.md                 # This documentation file.
```

### Running Tests

```bash
cd api
pytest -v
```

## API Endpoints

The `anomaly-detection-api` service (running on `http://localhost:8000`) exposes the following main endpoints:

*   **`GET /`**: Root endpoint. Returns the API status.
    *   Response: `{"status": "online", "service": "Anomaly Detection API", "version": "1.0.0"}`
*   **`GET /health`**: Health check endpoint. Returns the status of the API and its components (model, Redis).
    *   Response Example:
        ```json
        {
          "status": "healthy",
          "components": {
            "model": "available",
            "redis": "connected"
          },
          "metrics": {
            "transactions_processed": 150,
            "fraudulent_detected": 7
          }
        }
        ```
*   **`POST /predict`**: Allows manual submission of a single transaction for anomaly detection.
    *   Request Body (JSON):
        ```json
        {
          "user_id": "some-user-id",
          "amount": 123.45,
          "transaction_id": "manual-tx-001" // Optional
          // Other fields from the transaction schema can be included
        }
        ```
    *   Response Example:
        ```json
        {
          "transaction_id": "manual-tx-001",
          "is_anomalous": false,
          "anomaly_score": 0.0523,
          "features": {
            "amount": 123.45,
            "avg_amount": 110.0,
            // ... other features ...
          }
        }
        ```
*   **`GET /metrics`**: Exposes metrics in Prometheus format for scraping. (Typically accessed by Prometheus, not directly by users).

## Configuration

The system uses environment variables for configuration, primarily defined in the `docker-compose.yml` file for each service.

### Environment Variables

Key environment variables for the `anomaly-detection-api` service:

*   `KAFKA_BROKER`: Address of the Kafka broker (e.g., `kafka:29092`).
*   `REDIS_HOST`: Hostname for the Redis server (e.g., `redis`).
*   `REDIS_PORT`: Port for the Redis server (e.g., `6379`).
*   `MLFLOW_TRACKING_URI`: URI for the MLflow tracking server (e.g., `http://mlflow:5000`).

Key environment variables for the `producer` service:

*   `KAFKA_BROKER`: Address of the Kafka broker (e.g., `kafka:29092`).

Key environment variables for the `grafana` service:

*   `GF_SECURITY_ADMIN_USER`: Grafana admin username.
*   `GF_SECURITY_ADMIN_PASSWORD`: Grafana admin password.

## Monitoring

The Grafana dashboard (available at http://localhost:3000) provides real-time insights into the system's performance and anomaly detection process. Key panels include:

*   **Transaction Throughput**:
    *   Metric: `rate(transactions_processed_total[1m])`
    *   Description: Shows the number of transactions being processed by the API per second (averaged over 1 minute). Helps monitor the load and processing capacity.
*   **Transaction Types**:
    *   Metrics:
        *   `rate(fraudulent_transactions_total[1m])` (Fraudulent)
        *   `rate(transactions_processed_total[1m]) - rate(fraudulent_transactions_total[1m])` (Normal)
    *   Description: Displays the rate of transactions classified as fraudulent versus normal. Useful for understanding the prevalence of anomalies.
*   **Total Transactions**:
    *   Metric: `sum(transactions_processed_total)`
    *   Description: A counter showing the cumulative number of transactions processed since the API started.
*   **Total Fraud Detected**:
    *   Metric: `sum(fraudulent_transactions_total)`
    *   Description: A counter showing the cumulative number of transactions flagged as fraudulent.
*   **Processing Latency (p95, p50)**:
    *   Metrics:
        *   `histogram_quantile(0.95, sum(rate(transaction_processing_seconds_bucket[5m])) by (le))` (p95)
        *   `histogram_quantile(0.50, sum(rate(transaction_processing_seconds_bucket[5m])) by (le))` (p50)
    *   Description: Shows the 95th and 50th percentile of transaction processing time. This indicates how long it takes to process a transaction from feature extraction to prediction. p95 helps understand worst-case latency for most users.

## Troubleshooting

*   **Grafana Dashboards Not Loading Data**:
    *   Ensure Prometheus is running and successfully scraping the `anomaly-detection-api:8001/metrics` endpoint. Check Prometheus targets (`http://localhost:9090/targets`).
    *   Verify that the `anomaly-detection-api` is generating metrics. Check its logs for errors.
    *   Ensure the producer is sending data and the API is processing it.
*   **Model Not Loading in API**:
    *   Check `anomaly-detection-api` logs for errors related to MLflow or model loading.
    *   Ensure the MLflow service is running and accessible from the API container (`MLFLOW_TRACKING_URI`).
    *   Verify that a model named `anomaly_detector` is registered in MLflow and has a "latest" version. If not, run the training script.

## Future Improvements

- Add more complex anomaly detection models (e.g., deep learning-based)
- Implement model versioning and automatic retraining
- Add user authentication and authorization
- Scale horizontally with Kubernetes
- Implement A/B testing for model comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Data & Model Training Notice

Currently, the data used for model training is generated synthetically and does not reflect real-world transaction patterns. As a result, the trained anomaly detection model may not perform well on realistic or production data.

**How to improve:**
- Replace the synthetic data generation with data collected from a live API or real transaction logs.
- Retrain the model using features extracted from this real or simulated data for better anomaly detection performance.