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
    pkg-config \
    cmake \
    curl \
    git \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libsvtav1-dev \
    libaom-dev \
    libfdk-aac-dev \
    libass-dev \
    libbluray-dev \
    libdav1d-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    frei0r-plugins-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libopenjpeg-dev \
    libopus-dev \
    librav1e-dev \
    librist-dev \
    librubberband-dev \
    libshine-dev \
    libsnappy-dev \
    libsoxr-dev \
    libspeex-dev \
    libsrt-openssl-dev \
    libssh-dev \
    libtheora-dev \
    libvidstab-dev \
    libvorbis-dev \
    libwebp-dev \
    libxml2-dev \
    libxvidcore-dev \
    libzmq3-dev \
    libzimg-dev \
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
    --enable-pthreads \
    --enable-version3 \
    --enable-ffplay \
    --enable-gnutls \
    --enable-gpl \
    --enable-libaom \
    --enable-libaribb24 \
    --enable-libbluray \
    --enable-libdav1d \
    --enable-libharfbuzz \
    --enable-libjxl \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-librav1e \
    --enable-librist \
    --enable-librubberband \
    --enable-libsnappy \
    --enable-libsrt \
    --enable-libssh \
    --enable-libsvtav1 \
    --enable-libtesseract \
    --enable-libtheora \
    --enable-libvidstab \
    --enable-libvmaf \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libwebp \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libxml2 \
    --enable-libxvid \
    --enable-lzma \
    --enable-libfontconfig \
    --enable-libfreetype \
    --enable-frei0r \
    --enable-libass \
    --enable-libopencore-amrnb \
    --enable-libopencore-amrwb \
    --enable-libopenjpeg \
    --enable-libspeex \
    --enable-libsoxr \
    --enable-libzmq \
    --enable-libzimg \
    --disable-libjack \
    --disable-indev=jack \
    --enable-videotoolbox \
    --enable-audiotoolbox \
    --enable-neon \
    --cc=clang && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf /ffmpeg

# Copy Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir wheel && pip install --no-cache-dir -r /tmp/requirements.txt

# Set environment variable to use the new FFmpeg version
ENV PATH="/usr/local/bin:$PATH"