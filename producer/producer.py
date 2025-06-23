import json
import time
import os
import random
import csv
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

# Initialize Faker
fake = Faker()

# Configure Kafka producer
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TRANSACTIONS_TOPIC = "transactions"

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(0, 10, 0) # Ensure api_version is a tuple
)

# List of locations for variation
LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
]

# Create a pool of users for consistency in transactions
USERS = [
    {
        "user_id": fake.uuid4(),
        "spending_pattern": random.uniform(50, 2000),  # Base spending amount
        "volatility": random.uniform(0.1, 0.5)  # How much their spending varies
    }
    for _ in range(100)
]

def generate_transaction(add_anomaly=False):
    """Generate a fake financial transaction."""
    user = random.choice(USERS)
    user_id = user["user_id"]
    
    # Normal transaction amount based on user's spending pattern
    mean_amount = user["spending_pattern"]
    volatility = user["volatility"]
    
    if add_anomaly and random.random() < 0.1:  # 10% of anomalies are unusually large
        # Generate an anomalous transaction (significantly larger amount)
        amount = mean_amount * random.uniform(5.0, 10.0)
    else:
        # Generate a normal transaction amount
        amount = mean_amount * random.uniform(1 - volatility, 1 + volatility)
    
    transaction = {
        "transaction_id": fake.uuid4(),
        "user_id": user_id,
        "amount": round(amount, 2),
        "timestamp": datetime.now().isoformat(),
        "location": random.choice(LOCATIONS),
        # Additional features that could be useful for anomaly detection
        "merchant_category_code": random.randint(1000, 9999),
        "transaction_type": random.choice(["purchase", "withdrawal", "transfer", "payment"])
    }
    return transaction

def send_transaction(transaction):
    """Send transaction to Kafka topic."""
    try:
        producer.send(TRANSACTIONS_TOPIC, transaction)
        print(f"✅ Sent transaction: {transaction['transaction_id']} - Amount: ${transaction['amount']}")
    except Exception as e:
        print(f"❌ Failed to send transaction: {e}")

def save_transactions_to_csv(transactions, filename):
    """Save generated transactions to a CSV file."""
    if not transactions:
        return
    keys = transactions[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(transactions)

def main():
    """Main function to generate and send transactions."""
    print(f"🚀 Starting transaction producer, sending to {KAFKA_BROKER} topic {TRANSACTIONS_TOPIC}")
    
    transactions_count = 0
    anomalies_generated = 0
    transactions = []  # Collect transactions for saving
    
    while True:
        # Generate anomalies randomly (around 5% of transactions)
        add_anomaly = random.random() < 0.05
        
        transaction = generate_transaction(add_anomaly)
        send_transaction(transaction)
        transactions.append(transaction)
        
        transactions_count += 1
        if add_anomaly:
            anomalies_generated += 1
        
        # Logging stats periodically
        if transactions_count % 100 == 0:
            anomaly_rate = (anomalies_generated / transactions_count) * 100
            print(f"📊 Stats: Generated {transactions_count} transactions, {anomalies_generated} potential anomalies ({anomaly_rate:.2f}%)")
        
        # Save every 1000 transactions for training
        if transactions_count % 1000 == 0:
            save_transactions_to_csv(transactions, "producer_transactions.csv")
            transactions.clear()
        
        # Generate approximately 10 transactions per second
        time.sleep(0.1)

if __name__ == "__main__":
    # Wait for Kafka to be ready
    time.sleep(20)
    print("⚠️ Ensure the model is trained and registered in MLflow before starting the API service.")
    print("⚠️ If you see runs in the MLflow UI but no files in ./mlruns, check your MLflow artifact storage configuration.")
    main()