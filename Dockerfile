# Stage 1: Build Flask application
FROM python:3.9 AS builder

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Stage 2: Set up MySQL
FROM mysql:latest AS mysql_setup

# Set environment variables
ENV MYSQL_ROOT_PASSWORD=Kubernetes@1993
ENV MYSQL_USER=root
ENV MYSQL_PASSWORD=Kubernetes@1993
ENV MYSQL_DATABASE=crypto_coins

# Copy database initialization script
COPY init.sql /docker-entrypoint-initdb.d/

# Stage 3: Final image
FROM python:3.9-slim AS final

WORKDIR /app

# Copy Flask application files from builder stage
COPY --from=builder /app .

# Copy MySQL setup from mysql_setup stage
COPY --from=mysql_setup /usr/local/bin/mysql /usr/local/bin/
COPY --from=mysql_setup /usr/share/mysql /usr/share/mysql

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Expose port for Flask application
EXPOSE 5000

# Start the Flask application
CMD ["python", "main.py"]
