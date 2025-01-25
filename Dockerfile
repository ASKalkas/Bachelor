# Use the NVIDIA CUDA 12.0 runtime image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping \
    python3.10 \
    python3-pip \
    python3-dev \
    nano \
    git \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python packages with the appropriate index
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the application code
COPY . .

# Set the entry point
CMD ["sh", "-c", "python3 -u load_models.py && tail -f /dev/null"]