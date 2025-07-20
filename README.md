# BigDataLab3 - Secret Management with HashiCorp Vault

This project extends BigDataLab2 by adding secret management using HashiCorp Vault. Instead of storing database credentials in environment variables or configuration files, they are now securely stored in Vault.

## Architecture

The project consists of three Docker containers:
1. **Vault**: HashiCorp Vault for secret management
2. **Postgres**: PostgreSQL database for storing predictions
3. **App**: The machine learning model API

## Secret Management

Database credentials are stored in Vault and retrieved by the application at runtime. This improves security by:
- Encrypting secrets at rest
- Providing access control to secrets
- Centralizing secret management
- Enabling secret rotation

## Setup and Running

1. Clone the repository
2. Create a `.env` file based on `.env.example` (only needed for initial setup)
3. Run the application:
   ```
   docker-compose up -d
   ```

During the first run, Vault will be initialized and the database credentials will be stored in Vault. After that, the application will retrieve the credentials from Vault instead of using environment variables.

## API Endpoints

- `/predict`: Make a prediction based on a review
- `/predictions`: Get the latest predictions from the database
- `/health`: Check the health of the API, including Vault connection status
- `/vault-status`: Check the status of the Vault connection

## Security Notes

- The `.env` file is only used for initial setup and should be removed in production
- In production, Vault should be properly configured with appropriate authentication methods and policies
- The Vault token used by the application should be rotated regularly
