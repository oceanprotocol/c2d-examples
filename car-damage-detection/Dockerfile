FROM tensorflow/tensorflow:2.12.0-gpu

ENV DEBIAN_FRONTEND=noninteractive


COPY requirements.txt /tmp/
RUN pip install --upgrade pip wheel
RUN pip install -r /tmp/requirements.txt