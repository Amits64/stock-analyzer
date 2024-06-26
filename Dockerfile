# Stage 1: Build Stage
FROM tensorflow/tensorflow:latest-gpu AS build-stage

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install necessary packages and dependencies for building (if any)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --ignore-installed -r requirements.txt

# Stage 2: Production Stage
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the built artifacts from the build stage
COPY . .

# Install Python dependencies again in the production stage
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --ignore-installed -r requirements.txt

# Expose the port that Flask runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "main.py"]
