#!/bin/bash

# Update package index:
apt-get update
apt-get install \
ca-certificates \
curl \
wget \
g++ \
git \
ocl-icd-opencl-dev \
opencl-c-headers  \
pkg-config \
python \
python-dev \
python3.7 \
python3.7-dev \
python3-distutils \
unzip \
zip \
zlib1g-dev \
openjdk-11-jdk \
m4 \
libexempi-dev \
rsync \
python3-numpy \
build-essential \
libsdl2-dev \
libjpeg-dev \
nasm \
tar \
libbz2-dev \
libgtk2.0-dev \
cmake \
libfluidsynth-dev \
libgme-dev \
libopenal-dev \
timidity \
libwildmidi-dev \
libboost-all-dev \
libsdl2-dev \
mysql-server \
libmysqlclient-dev \

# Install bazel:
curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh && bash /tmp/bazel.sh && rm /tmp/bazel.sh

python3 -m pip install 'pybind11==2.4.3'
