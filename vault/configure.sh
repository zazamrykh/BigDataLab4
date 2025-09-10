#!/bin/sh
set -e

# Wait for Vault to start
sleep 5

# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'

# In dev mode, the root token is always "root"
export VAULT_TOKEN="root"

# Check if Vault is already initialized
INIT_STATUS=$(vault status -format=json 2>/dev/null | jq -r '.initialized')

if [ "$INIT_STATUS" = "true" ]; then
  echo "Vault is already initialized. Using dev mode configuration."

  # In dev mode, Vault is already unsealed
  # We just need to update the secrets

  echo "Updating database credentials in Vault..."
  vault kv put secret/database/credentials \
    username="${DB_USER}" \
    password="${DB_PASSWORD}" \
    dbname="${DB_NAME}" \
    port="${DB_PORT}" \
    host=postgres

  echo "Updating Kafka credentials in Vault..."
  vault kv put secret/kafka/credentials \
    bootstrap_servers=kafka:9092

  echo "Vault configuration updated!"

  # Display the secrets to verify they were saved correctly
  echo "========== Secret Path =========="
  vault kv get secret/database/credentials
  echo "========== Secret Path =========="
  vault kv get secret/kafka/credentials
else
  echo "Vault is not initialized. This should not happen in dev mode."
  exit 1
fi
