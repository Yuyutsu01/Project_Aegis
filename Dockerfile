# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Pre-install CPU version of PyTorch to avoid massive CUDA downloads (reduces image size significantly)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the entrypoint script which executes the pipeline and launches Streamlit
CMD ["python", "entrypoint.py"]
