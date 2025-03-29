FROM python:3.8

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3-tk \
    tk-dev \
    python3-pip \
    libagg-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0
    
COPY requirements.txt /tmp/
RUN pip install wheel
RUN pip install -r /tmp/requirements.txt