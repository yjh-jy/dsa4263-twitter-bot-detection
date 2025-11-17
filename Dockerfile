FROM python:3.10.6

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip; \
    pip install --no-cache-dir -r requirements.txt; \
    pip install --no-cache-dir jupyterlab

# Copy project
COPY . .

# Ensure runtime directories exist
RUN mkdir -p \
      /app/data \
      /app/models \
      /app/reports \
      /app/notebooks

# These will be mounted as volumes in docker-compose
VOLUME ["/app/data", "/app/models", "/app/reports", "/app/image_cache"]

# # Security — non-root user
# RUN useradd -ms /bin/bash appuser && chown -R appuser /app
# USER appuser

# Exposed port (used for Jupyter)
EXPOSE 8888

# Default command (overridden by docker compose if needed)
CMD ["bash"]
