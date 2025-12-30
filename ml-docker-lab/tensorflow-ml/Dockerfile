FROM tensorflow/tensorflow:2.13.0

# Set working directory
WORKDIR /app

# Install additional Python packages
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0

# Copy application files
COPY . /app

# Set default command
CMD ["python3"]
