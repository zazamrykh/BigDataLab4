#!/bin/sh
set -e

# Wait for Vault to start
sleep 5

# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'  # использовать localhost внутри контейнера [1]

# In dev mode, the root token is always "root"
export VAULT_TOKEN="root"

# Check if Vault is already initialized
INIT_STATUS=$(vault status -format=json 2>/dev/null | jq -r '.initialized')

if [ "$INIT_STATUS" = "true" ]; then
  echo "Vault is already initialized. Using dev mode configuration."

<<<<<<< HEAD
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
    bootstrap_servers=kafka:${KAFKA_PORT:-9092}

  echo "Vault configuration updated!"

  # Display the secrets to verify they were saved correctly
  echo "========== Secret Path =========="
  vault kv get secret/database/credentials
  echo "========== Secret Path =========="
  vault kv get secret/kafka/credentials
=======
  if [ -f /vault/data/root_token.txt ]; then
    VAULT_ROOT_TOKEN=$(cat /vault/data/root_token.txt)

    # Проверка и разгерметизация при необходимости
    SEAL_STATUS=$(vault status -format=json 2>/dev/null | jq -r '.sealed')
    if [ "$SEAL_STATUS" = "true" ] && [ -f /vault/data/unseal_key.txt ]; then
      VAULT_UNSEAL_KEY=$(cat /vault/data/unseal_key.txt)
      echo "Unsealing Vault..."
      vault operator unseal "$VAULT_UNSEAL_KEY"
    fi

    # Без интерактивного login: просто экспортируем токен
    export VAULT_TOKEN="$VAULT_ROOT_TOKEN"  # ключевая правка вместо 'vault login' [1]

    # Check if environment variables are set
    if [ -z "${DB_USER}" ] || [ -z "${DB_PASSWORD}" ] || [ -z "${DB_NAME}" ] || [ -z "${DB_PORT}" ]; then
      echo "Warning: One or more database environment variables are not set."
      echo "DB_USER: ${DB_USER:-not set}"
      echo "DB_PASSWORD: ${DB_PASSWORD:-not set}"
      echo "DB_NAME: ${DB_NAME:-not set}"
      echo "DB_PORT: ${DB_PORT:-not set}"
      echo "Using default values for missing variables."

      # Set default values if not provided
      DB_USER=${DB_USER:-postgres}
      DB_PASSWORD=${DB_PASSWORD:-postgres}
      DB_NAME=${DB_NAME:-reviewdb}
      DB_PORT=${DB_PORT:-5432}
    fi

    echo "Updating database credentials in Vault..."
    vault kv put kv/database/credentials \
      username="${DB_USER}" \
      password="${DB_PASSWORD}" \
      dbname="${DB_NAME}" \
      port="${DB_PORT}" \
      host=postgres  # запись под root‑токеном [2]

    echo "Updating Kafka credentials in Vault..."
    vault kv put kv/kafka/credentials \
      bootstrap_servers=kafka:9092  # запись под root‑токеном [2]

    echo "Vault configuration updated!"
  else
    echo "Root token not found. Cannot configure Vault."
    exit 1
  fi
>>>>>>> upstream/main
else
  echo "Vault is not initialized. This should not happen in dev mode."
  exit 1
fi
