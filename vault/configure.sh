#!/bin/sh

# Wait for Vault to start
sleep 5

# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'

# Check if Vault is already initialized
INIT_STATUS=$(vault status -format=json 2>/dev/null | jq -r '.initialized')

if [ "$INIT_STATUS" = "true" ]; then
  echo "Vault is already initialized. Using existing configuration."

  # Check if we have a root token
  if [ -f /vault/data/root_token.txt ]; then
    VAULT_ROOT_TOKEN=$(cat /vault/data/root_token.txt)

    # Check if Vault is sealed
    SEAL_STATUS=$(vault status -format=json 2>/dev/null | jq -r '.sealed')

    if [ "$SEAL_STATUS" = "true" ] && [ -f /vault/data/unseal_key.txt ]; then
      VAULT_UNSEAL_KEY=$(cat /vault/data/unseal_key.txt)
      echo "Unsealing Vault..."
      vault operator unseal $VAULT_UNSEAL_KEY
    fi

    # Login with root token
    echo "Logging in with root token..."
    vault login $VAULT_ROOT_TOKEN

    # Update database credentials in Vault
    echo "Updating database credentials in Vault..."
    vault kv put kv/database/credentials \
      username=${DB_USER} \
      password=${DB_PASSWORD} \
      dbname=${DB_NAME} \
      port=${DB_PORT} \
      host=postgres

    # Update Kafka credentials in Vault
    echo "Updating Kafka credentials in Vault..."
    vault kv put kv/kafka/credentials \
      bootstrap_servers=kafka:9092

    echo "Vault configuration updated!"
  else
    echo "Root token not found. Cannot configure Vault."
    exit 1
  fi
else
  echo "Vault is not initialized. Please run init.sh first."
  exit 1
fi
