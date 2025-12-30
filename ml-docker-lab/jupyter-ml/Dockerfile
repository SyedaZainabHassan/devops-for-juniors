FROM jupyter/tensorflow-notebook:latest

# Switch to root user to install packages
USER root

# Install additional system packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user
USER jovyan

# Install additional Python packages
RUN pip install --no-cache-dir \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    scikit-learn==1.3.0 \
    xgboost==1.7.6

# Create notebooks directory
RUN mkdir -p /home/jovyan/notebooks

# Copy notebook files
COPY notebooks/ /home/jovyan/notebooks/

# Set working directory
WORKDIR /home/jovyan/notebooks

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
