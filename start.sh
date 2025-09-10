#!/bin/bash

# Script to start the application with environment variables passed as parameters
# This script launches services separately to ensure proper isolation of environment variables

# Default values
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_NAME="reviewdb"
DB_PORT="5432"
VAULT_ADDR="http://vault:8200"
KAFKA_PORT="9092"

# Default flag for cleaning volumes
CLEAN_VOLUMES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --db-user=*)
      DB_USER="${1#*=}"
      shift
      ;;
    --db-password=*)
      DB_PASSWORD="${1#*=}"
      shift
      ;;
    --db-name=*)
      DB_NAME="${1#*=}"
      shift
      ;;
    --db-port=*)
      DB_PORT="${1#*=}"
      shift
      ;;
    --vault-addr=*)
      VAULT_ADDR="${1#*=}"
      shift
      ;;
    --kafka-port=*)
      KAFKA_PORT="${1#*=}"
      shift
      ;;
    --env-file=*)
      ENV_FILE="${1#*=}"
      shift
      ;;
    --clean)
      CLEAN_VOLUMES=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --db-user=USER         Database user (default: postgres)"
      echo "  --db-password=PASSWORD Database password (default: postgres)"
      echo "  --db-name=NAME         Database name (default: reviewdb)"
      echo "  --db-port=PORT         Database port (default: 5432)"
      echo "  --vault-addr=ADDR      Vault address (default: http://vault:8200)"
      echo "  --kafka-port=PORT      Kafka port (default: 9092)"
      echo "  --env-file=FILE        Environment file to use"
      echo "  --clean                Remove existing volumes before starting (use with caution!)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# If env file is provided, use it
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  echo "Using environment file: $ENV_FILE"
  # Start docker-compose with env file
  docker compose --env-file "$ENV_FILE" up -d
  echo "Application started successfully with environment file!"
else
  echo "Starting services separately with appropriate environment variables..."

  # Clean volumes if requested
  if [ "$CLEAN_VOLUMES" = true ]; then
    echo "Cleaning existing volumes..."
    docker compose down -v
    echo "Volumes removed."
  fi

  # Step 1: Start Vault with database credentials
  # Vault needs DB credentials to store them in the vault
  echo "Starting Vault service..."
  DB_USER=$DB_USER \
  DB_PASSWORD=$DB_PASSWORD \
  DB_NAME=$DB_NAME \
  DB_PORT=$DB_PORT \
  docker compose up -d vault

  # Wait for Vault to be ready
  echo "Waiting for Vault to be ready..."
  for i in {1..30}; do
    if curl -s http://localhost:8200/v1/sys/health | grep -q '"initialized":true'; then
      echo "Vault is ready!"
      break
    fi
    echo "Waiting for Vault to initialize... ($i/30)"
    sleep 2
    if [ $i -eq 30 ]; then
      echo "Vault failed to initialize in time. Check logs with: docker compose logs vault"
      exit 1
    fi
  done

  # Step 2: Start PostgreSQL with initialization variables
  # PostgreSQL needs these variables only for initialization
  echo "Starting PostgreSQL service..."
  # Export variables so they are available to docker-compose
  export POSTGRES_USER=$DB_USER
  export POSTGRES_PASSWORD=$DB_PASSWORD
  export POSTGRES_DB=$DB_NAME
  docker compose up -d postgres

  # Wait for PostgreSQL to be ready
  echo "Waiting for PostgreSQL to be ready..."
  for i in {1..30}; do
    if docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; then
      echo "PostgreSQL is ready!"
      break
    fi
    echo "Waiting for PostgreSQL to initialize... ($i/30)"
    sleep 2
    if [ $i -eq 30 ]; then
      echo "PostgreSQL failed to initialize in time. Check logs with: docker compose logs postgres"
      exit 1
    fi
  done

  # Step 3: Start Kafka
  echo "Starting Kafka service..."
  # Export Kafka port
  export KAFKA_PORT
  docker compose up -d kafka

  # Wait for Kafka to be ready
  echo "Waiting for Kafka to be ready..."
  for i in {1..30}; do
    if docker compose exec kafka bash -c "echo > /dev/tcp/localhost/$KAFKA_PORT" > /dev/null 2>&1; then
      echo "Kafka is ready!"
      break
    fi
    echo "Waiting for Kafka to initialize... ($i/30)"
    sleep 2
    if [ $i -eq 30 ]; then
      echo "Kafka failed to initialize in time. Check logs with: docker compose logs kafka"
      exit 1
    fi
  done

  # Step 4: Start the application without database credentials
  # App will get credentials from Vault
  echo "Starting application service..."
  docker compose up -d app

  # Step 5: Start the database consumer
  # Export DB variables for db-consumer as fallback
  echo "Starting database consumer service..."
  # Export variables so they are available to docker-compose
  export DB_HOST=postgres
  export DB_PORT
  export DB_NAME
  export DB_USER
  export DB_PASSWORD
  docker compose up -d db-consumer

  echo "All services started successfully!"
fi

echo "You can access the API at http://localhost:8000"
echo "You can access Vault at http://localhost:8200"
