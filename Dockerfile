FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies - with the exact dependencies recommended by the library author
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpoppler-cpp-dev \
        pkg-config \
        ocrmypdf \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        libmagic1 \
        qpdf \
        git \
        curl && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs data/vector_db

# Expose ports for both services
EXPOSE 8000 8501

# Default command runs the MCP server
# (docker-compose overrides this for the Streamlit container)
CMD ["python", "src/model_context_protocol/card_data_server.py"]
