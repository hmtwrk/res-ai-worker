# Use a base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system tools (ffmpeg is often needed for audio)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY handler.py .

# Start the worker
CMD [ "python", "-u", "handler.py" ]
