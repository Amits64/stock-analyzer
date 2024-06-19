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
RUN pip install -r requirements.txt

# Stage 2: Production Stage
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the built artifacts from the build stage
COPY --from=build-stage /app /app

# Install only runtime dependencies (if necessary)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Expose the port that Flask runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "main.py"]
