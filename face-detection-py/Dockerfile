FROM python:3.8

RUN apt-get update && \
    apt-get install -y \
    python3-tk \
    tk-dev \
    clang \
    python3-pip \
    libagg-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg=7:7.1.1-1ubuntu1 \
    && apt-get clean

COPY requirements.txt /tmp/
RUN pip install wheel
RUN pip install -r /tmp/requirements.txt