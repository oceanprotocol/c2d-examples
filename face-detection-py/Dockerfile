FROM python:3.8

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-tk \
    tk-dev \
    python3-pip \
    libagg-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    clang \
    libc++-dev \
    libc++abi-dev \
    ninja-build \
    nasm \
    yasm \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libsvtav1-dev \
    libfdk-aac-dev \
    pkg-config \
    cmake \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Clang as default compiler
ENV CC=clang
ENV CXX=clang++

# Build and Install FFmpeg (latest version)
RUN git clone --depth=1 https://git.ffmpeg.org/ffmpeg.git /ffmpeg && \
    cd /ffmpeg && \
    ./configure \
    --prefix=/usr/local \
    --disable-debug \
    --disable-doc \
    --disable-static \
    --enable-shared \
    --enable-gpl \
    --enable-version3 \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libsvtav1 \
    --enable-libfdk-aac \
    --enable-pic \
    --cc=clang && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf /ffmpeg

# Copy Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir wheel && pip install --no-cache-dir -r /tmp/requirements.txt

# Set environment variable to use the new FFmpeg version
ENV PATH="/usr/local/bin:$PATH"