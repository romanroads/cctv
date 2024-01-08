FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

MAINTAINER Joe Xing <joe.xing@romanroads.io>

ENV PATH /opt/conda/bin:$PATH

RUN set -xe \
    && apt-get update \
    && apt-get -y install gcc g++ git ca-certificates \
    && apt-get clean

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git libpq-dev && \
    apt-get clean && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
     echo "conda activate base" >> ~/.bashrc

COPY . /cctv

WORKDIR /cctv/python

RUN /bin/bash -c 'conda env create -f environment_ubuntu.yml && . activate element_ubuntu'

RUN /bin/bash -c 'ln -s /cctv /home/element_cctv_symbolic_link'

WORKDIR /cctv/