# Docker Setup for Credit Card Rewards Optimizer

This document describes how to use Docker to run the Credit Card Rewards Optimizer application, consisting of the card_data_server (MCP server) and the Streamlit frontend app.

## Prerequisites

- Docker and Docker Compose installed on your system
- An OpenAI API key (to be placed in the `.env` file)

## Getting Started

1. Make sure you have a valid `.env` file with your OpenAI API key:

    ```bash
    # Copy the example file
    cp .env.example .env

    # Edit the .env file and add your OpenAI API key
    nano .env  # or use any text editor
    ```

2. Build and start the Docker containers:

    ```bash
    docker-compose up -d
    ```

    This will start both services:

    - The MCP server on port 8000
    - The Streamlit app on port 8501

3. Access the application:
   - Streamlit web interface: [http://localhost:8501](http://localhost:8501)
   - MCP server endpoint: [http://localhost:8000](http://localhost:8000)

4. To view the logs:

    ```bash
    # View logs for both services
    docker-compose logs -f

    # View logs for a specific service
    docker-compose logs -f card-data-server
    docker-compose logs -f streamlit-app
    ```

5. To stop the containers:

    ```bash
    docker-compose down
    ```

## Configuration

You can modify the environment variables in the `.env` file to customize your deployment:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: The OpenAI model to use (default: gpt-4o-2024-08-06)
- `VECTOR_DB_TYPE`: The vector database type (default: chroma)
- `VECTOR_DB_PATH`: The path to the vector database (default: ./data/vector_db)
- `DEBUG_MODE`: Whether to enable debug mode (default: false)
- `STREAMLIT_THEME_BASE`: Light or dark theme for Streamlit (default: light)

## Data Persistence

The Docker setup mounts the following directories as volumes:

- `./data`: Contains the vector database and other data files
- `./logs`: Contains the application logs

This ensures that your data persists even if the containers are removed.

## Running Services Individually

If you need to run only one of the services:

```bash
# Run only the MCP server
docker-compose up -d card-data-server

# Run only the Streamlit app
docker-compose up -d streamlit-app
```

## Building the Docker Image Directly

If you prefer to build the Docker image directly without Docker Compose:

```bash
# Build the image
docker build -t credit-card-optimizer .

# Run the MCP server
docker run -d --name card-data-server \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  credit-card-optimizer

# Run the Streamlit app
docker run -d --name streamlit-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  credit-card-optimizer python app.py
```

## Troubleshooting

If you encounter issues:

1. Check the container logs:

   ```bash
   docker-compose logs
   ```

2. Ensure your `.env` file contains a valid OpenAI API key

3. Make sure the required data directories exist:

   ```bash
   mkdir -p data/vector_db logs
   ```

4. If the Streamlit app can't connect to the MCP server, ensure that:
   - The MCP server container is running
   - The app is configured to connect to the correct MCP server endpoint

5. If services aren't responding, try restarting the containers:

   ```bash
   docker-compose restart
   ```
