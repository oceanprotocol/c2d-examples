FROM node:20-alpine

ENV NPM_CONFIG_PREFIX=/home/node/.npm-global \
    PATH=$PATH:/home/node/.npm-global/bin \
    NODE_PATH=/home/node/.npm-global/lib/node_modules/ \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/venv \
    PATH="/venv/bin:$PATH"

RUN apk update && \
    apk add --no-cache \
        ffmpeg \
        python3 \
        py3-pip \
        py3-virtualenv \
        pkgconfig \
        g++ \
        jpeg-dev \
        cairo-dev \
        giflib-dev \
        pango-dev \
        libtool \
        autoconf \
        automake \
        libpng \
        libpng-dev \
        build-base \
        ca-certificates \
        wget && \
    ln -sf python3 /usr/bin/python && \
    python3 -m venv /venv && \
    /venv/bin/pip install --no-cache --upgrade pip setuptools

WORKDIR /usr/app
COPY . /usr/app

RUN npm install && \
    npm cache clean --force