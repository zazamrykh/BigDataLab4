#!/bin/bash

# Test script for Kafka integration
echo "Testing Kafka integration..."

# Check if containers are running
echo "Checking if containers are running..."
if ! docker ps | grep -q kafka; then
  echo "Kafka container is not running. Starting containers..."
  docker-compose up -d

  # Wait for containers to start
  echo "Waiting for containers to start..."
  sleep 30
fi

# Check Kafka status
echo "Checking Kafka status..."
KAFKA_STATUS=$(curl -s http://localhost:8000/kafka-status || echo '{"connected":false}')
if echo $KAFKA_STATUS | grep -q '"connected":true'; then
  echo "Kafka is connected."
else
  echo "Kafka is not connected. Please check the logs."
  docker-compose logs kafka
  exit 1
fi

# Check API health
echo "Checking API health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health || echo '{"status":"unhealthy"}')
if echo $HEALTH_RESPONSE | grep -q '"status":"healthy"'; then
  echo "API is healthy."
else
  echo "API is not healthy. Please check the logs."
  docker-compose logs app
  exit 1
fi

# Check Kafka connection in API
echo "Checking Kafka connection in API..."
if echo $HEALTH_RESPONSE | grep -q '"kafka_connected":true'; then
  echo "API is connected to Kafka."
else
  echo "API is not connected to Kafka. Please check the logs."
  docker-compose logs app
  exit 1
fi

# Make a prediction
echo "Making a prediction..."
PREDICT_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"summary": "Great product!", "text": "This product works perfectly and I love it.", "HelpfulnessNumerator": 5, "HelpfulnessDenominator": 7}')

if echo $PREDICT_RESPONSE | grep -q '"prediction":'; then
  echo "Prediction successful."
else
  echo "Prediction failed. Please check the logs."
  docker-compose logs app
  exit 1
fi

# Wait for consumer to process message
echo "Waiting for consumer to process message..."
sleep 5  # Wait for the consumer to process the message

# Check logs from db-consumer to verify message was processed
DB_CONSUMER_LOGS=$(docker-compose logs db-consumer)
echo "DB Consumer logs: $DB_CONSUMER_LOGS"

if echo "$DB_CONSUMER_LOGS" | grep -q "Successfully saved prediction to database"; then
  echo "Message successfully processed by consumer!"
else
  echo "Message not processed by consumer!"
  docker-compose logs
  exit 1
fi

# Check if prediction was saved in database
echo "Checking if prediction was saved in database..."
# This would require connecting to the database directly
# For simplicity, we rely on the consumer logs to verify this

echo "All tests passed! Kafka integration is working correctly."
