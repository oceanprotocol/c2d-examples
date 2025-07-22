FROM tensorflow/tensorflow:2.12.0-gpu

ENV DEBIAN_FRONTEND=noninteractive


COPY requirements.txt /tmp/
RUN pip install --upgrade pip wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt