
FROM --platform=linux/amd64 python:3.8

# Set working directory
WORKDIR /fairmot

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-distutils \
    python3-setuptools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    liblapack-dev \
    libblas-dev \
    libatlas-base-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/* 
    
    
# Install pip packages
COPY FairMOT/requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install setuptools==59.6.0
RUN pip install --no-cache-dir cython numpy==1.23.0
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install torch==1.7.0+cpu torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git \
    && cd DCNv2 \
    && ./make.sh

COPY . .


CMD ["/bin/bash"]
