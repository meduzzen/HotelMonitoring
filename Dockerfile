# Use official Python 3.10 base image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and video writing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scipy==1.16.0 \
    && pip install --no-cache-dir -U numpy

RUN sed -i '/checkpoint = torch.load(fpath, map_location=map_location)/s//checkpoint = torch.load(fpath, map_location=map_location, weights_only=False)/' /usr/local/lib/python3.11/site-packages/torchreid/reid/utils/torchtools.py

# Upgrade pip and install Python dependencies
# Copy your source code
COPY . .

# Run the app
CMD ["python", "main.py"]
