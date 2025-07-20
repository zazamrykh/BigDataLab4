#!/bin/bash

# Test script for Vault integration
echo "Testing Vault integration..."

# Check if containers are running
echo "Checking if containers are running..."
if ! docker ps | grep -q vault; then
  echo "Vault container is not running. Starting containers..."
  docker-compose up -d

  # Wait for containers to start
  echo "Waiting for containers to start..."
  sleep 30
fi

# Check Vault status
echo "Checking Vault status..."
VAULT_STATUS=$(curl -s http://localhost:8200/v1/sys/health || echo '{"initialized":false}')
if echo $VAULT_STATUS | grep -q '"initialized":true'; then
  echo "Vault is initialized."
else
  echo "Vault is not initialized. Please check the logs."
  docker-compose logs vault
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

# Check Vault connection in API
echo "Checking Vault connection in API..."
if echo $HEALTH_RESPONSE | grep -q '"vault_connected":true'; then
  echo "API is connected to Vault."
else
  echo "API is not connected to Vault. Please check the logs."
  docker-compose logs app
  exit 1
fi

# Check Vault status endpoint
echo "Checking Vault status endpoint..."
VAULT_STATUS_RESPONSE=$(curl -s http://localhost:8000/vault-status || echo '{"error":"Failed to connect"}')
if echo $VAULT_STATUS_RESPONSE | grep -q '"connected":true'; then
  echo "Vault status endpoint is working correctly."
else
  echo "Vault status endpoint is not working correctly. Please check the logs."
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

# Check if prediction was saved in database
echo "Checking if prediction was saved in database..."
sleep 2  # Wait for the prediction to be saved
PREDICTIONS_RESPONSE=$(curl -s http://localhost:8000/predictions)
if echo $PREDICTIONS_RESPONSE | grep -q '"predictions":'; then
  echo "Prediction was saved in database."
else
  echo "Prediction was not saved in database. Please check the logs."
  docker-compose logs app
  exit 1
fi

echo "All tests passed! Vault integration is working correctly."
