# ==============================================================================
# MERTFORMER TITAN - PRODUCTION CONTAINER (V27.0)
# ==============================================================================
# Base: PyTorch 2.6 + CUDA 12.4 + cuDNN 9 (Devel for compilation)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Metadata
LABEL maintainer="MertFormer Team"
LABEL version="27.0-FINAL"
LABEL description="MertFormer Titan Production Training Environment"

# Set working directory
WORKDIR /app

# Install System Dependencies (Ninja is critical for Flash Attention build)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-Install Flash Attention 2 (Time Saver: ~10 mins)
# We do this at build time so runtime is instant.
RUN pip install flash-attn --no-build-isolation

# Copy Complete Project
COPY . .

# Ensure scripts are executable
RUN chmod +x run.sh

# Set Environment Variables for Optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OMP_NUM_THREADS=8

# Entrypoint: The Ultimate Launchpad
ENTRYPOINT ["/bin/bash", "run.sh"]
