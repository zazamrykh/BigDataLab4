#!/bin/sh

# Start Vault server in dev mode
vault server -dev -dev-root-token-id=root -dev-listen-address=0.0.0.0:8200 &

# Wait for Vault to start
sleep 5

# In dev mode, Vault is already initialized and unsealed
# We just need to configure it
echo "Vault started in dev mode. Configuring..."

# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='root'

# In dev mode, KV v2 is already enabled at 'secret/'
# We don't need to enable it again

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

# Store database credentials in Vault
# For KV v2, we use 'kv put'
vault kv put secret/database/credentials \
  username="${DB_USER}" \
  password="${DB_PASSWORD}" \
  dbname="${DB_NAME}" \
  port="${DB_PORT}" \
  host=postgres

# Store Kafka credentials in Vault
vault kv put secret/kafka/credentials \
  bootstrap_servers=kafka:${KAFKA_PORT:-9092}

echo "Vault has been configured!"
echo "========== Secret Path =========="
vault kv get secret/database/credentials
echo "========== Secret Path =========="
vault kv get secret/kafka/credentials

# Keep the container running
tail -f /dev/null
