FROM python:3.7

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3-tk \
    tk-dev \
    libagg-dev
    
#COPY requirements.txt /tmp/
#RUN pip install wheel
#RUN pip install -r /tmp/requirements.txt
ENV THREADS=8
ENV SECONDS=360
ENV MEMORY=1024
CMD ["python3","cpumem-test.py"]