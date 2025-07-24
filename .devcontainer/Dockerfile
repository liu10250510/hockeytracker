FROM python:3.10-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        gcc \
        g++ \
        cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/vscode/.local/bin:$PATH"

# Create vscode user with sudo
RUN useradd -ms /bin/bash vscode \
    && mkdir -p /home/vscode/.vscode-server/extensions \
    && chown -R vscode:vscode /home/vscode

# Set the working directory
WORKDIR /workspace

# Create directories for data, models, etc.
RUN mkdir -p /workspace/data /workspace/models /workspace/notebooks

# Copy requirements.txt file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install wheel setuptools \
    && pip install -r /tmp/requirements.txt \
    && pip install torch torchvision torchaudio \
    && pip install black pylint isort pytest coverage

# Switch to the vscode user
USER vscode

# Set up a working directory
WORKDIR /workspace
