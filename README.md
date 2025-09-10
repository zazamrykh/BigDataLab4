# BigDataLab4 - Apache Kafka Integration

This project extends BigDataLab3 by adding message queue integration using Apache Kafka. Instead of directly saving predictions to the database, the API now sends messages to Kafka, which are then consumed by a dedicated service that saves them to the database.

## Architecture

The project consists of five Docker containers:
1. **Vault**: HashiCorp Vault for secret management
2. **Postgres**: PostgreSQL database for storing predictions
3. **Kafka**: Message broker for asynchronous communication
4. **App**: The machine learning model API (Producer)
5. **DB-Consumer**: Service that consumes messages from Kafka and saves them to the database

## Secret Management

Database and Kafka credentials are stored in Vault and retrieved by the applications at runtime. This improves security by:
- Encrypting secrets at rest
- Providing access control to secrets
- Centralizing secret management
- Enabling secret rotation

## Message Queue with Kafka

The project uses Apache Kafka as a message broker to implement a publish-subscribe pattern:
- The API service (Producer) publishes prediction results to a Kafka topic
- The DB-Consumer service subscribes to this topic and processes the messages
- This architecture provides:
  - Decoupling of services
  - Asynchronous processing
  - Better scalability
  - Improved fault tolerance

## Setup and Running

1. Clone the repository
2. Run the application using the provided script with parameters:
   ```
   ./start.sh --db-user=postgres --db-password=postgres --db-name=reviewdb --db-port=5432 --kafka-port=9092 --clean
   ```

   Alternatively, you can use an environment file:
   ```
   ./start.sh --env-file=.env
   ```

The script will:
1. Start Vault with database credentials
2. Start PostgreSQL with initialization variables
3. Start Kafka
4. Start the application without database credentials (it will retrieve them from Vault)
5. Start the database consumer

This approach ensures that database credentials are only passed to the services that need them, and the application retrieves credentials only from Vault.

### Command Line Options

```
Usage: ./start.sh [options]
Options:
  --db-user=USER         Database user (default: postgres)
  --db-password=PASSWORD Database password (default: postgres)
  --db-name=NAME         Database name (default: reviewdb)
  --db-port=PORT         Database port (default: 5432)
  --vault-addr=ADDR      Vault address (default: http://vault:8200)
  --kafka-port=PORT      Kafka port (default: 9092)
  --env-file=FILE        Environment file to use
  --clean                Remove existing volumes before starting (use with caution!)
  --help                 Show this help message
```

> **Note**: If you encounter issues with PostgreSQL authentication or "role does not exist" errors, try using the `--clean` flag to remove existing volumes and start fresh. This is especially useful when changing database credentials.

## API Endpoints

- `/predict`: Make a prediction based on a review (sends result to Kafka)
- `/health`: Check the health of the API, including Vault and Kafka connection status
- `/vault-status`: Check the status of the Vault connection
- `/kafka-status`: Check the status of the Kafka connection

## Testing

To test the Kafka integration, you can run the provided test script:
```
./test_kafka_integration.sh
```

This script will:
1. Check if the containers are running
2. Verify the Kafka connection
3. Make a prediction request
4. Verify that the message was processed by the consumer
5. Check that the prediction was saved to the database

## Security Notes

- The `.env` file is only used for initial setup and should be removed in production
- In production, Vault should be properly configured with appropriate authentication methods and policies
- The Vault token used by the applications should be rotated regularly
- Kafka should be properly secured in production with authentication and encryption


## Test scenario

- sudo docker compose down
- sudo docker compose up -d (--build)
- sudo docker compose ps
- curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "This product is terrible!", "summary": "Product review"}'
- sudo docker compose -f docker-compose.yml logs --tail=20 app
- sudo docker compose -f docker-compose.yml logs --tail=20 db-consumer
