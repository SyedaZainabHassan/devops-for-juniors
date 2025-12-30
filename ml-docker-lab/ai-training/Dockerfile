FROM tensorflow/tensorflow:2.13.0-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for AI/ML
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    opencv-python==4.8.0.74 \
    Pillow==10.0.0 \
    requests==2.31.0 \
    tqdm==4.65.0 \
    wandb==0.15.8

# Create directories for data, models, and logs
RUN mkdir -p /app/data /app/models /app/logs /app/scripts

# Copy training scripts
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Set permissions
RUN chmod +x /app/scripts/*.py

# Default command
CMD ["python3", "/app/scripts/train_model.py"]
