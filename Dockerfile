# Use official Python runtime as a parent image
FROM python:3.8-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory (if it doesn't exist)
RUN mkdir -p static/uploads

# Make port 8080 available (adjust if necessary)
EXPOSE 8080

# Specify the command to run the application
CMD ["python", "main.py"]
