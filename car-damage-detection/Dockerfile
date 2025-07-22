FROM tensorflow/tensorflow:2.12.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --upgrade pip wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt