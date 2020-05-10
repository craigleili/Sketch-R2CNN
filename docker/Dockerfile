FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  software-properties-common
RUN add-apt-repository ppa:jeff250/opencv

RUN apt-get update && apt-get install -y --no-install-recommends \
  cmake \
  curl \
  git \
  libgtk2.0-0 \
  libjpeg-dev \
  liblmdb-dev \
  libopenblas-dev \
  libopencv-contrib-dev \
  libopencv-dev \
  libopencv-nonfree-dev \
  libopenmpi-dev \
  libpng-dev \
  libsm6 \
  libxext6 \
  ninja-build \
  openmpi-bin \
  openmpi-doc \
  pkg-config \
  rsync \
  unzip \
  vim \
  wget \
  zlib1g \
  zlib1g-dev \
  imagemagick && \
  rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz -O ~/hdf5.tar.gz && \
  tar -xvzf ~/hdf5.tar.gz -C ~/ && cd ~/hdf5-1.10.5 && \
  ./configure --enable-parallel --enable-shared --prefix=/usr/local/hdf5 && \
  make -j 4 && make install -j 4 && \
  cd ~ && rm -rf ~/hdf5-1.10.5 && rm ~/hdf5.tar.gz

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh

RUN /opt/conda/bin/conda install -c anaconda protobuf
RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install --upgrade --ignore-installed setuptools
RUN /opt/conda/bin/pip install numpy scipy scikit-image matplotlib pytz PyYAML Pillow tqdm protobuf ninja lmdb addict fire rdp opencv-python typing mpi4py
RUN /opt/conda/bin/pip install torch==1.0.1.post2 torchvision==0.2.2.post3 pytorch-ignite==0.2.0 tensorboardX==1.7
RUN /opt/conda/bin/pip install git+https://github.com/rbgirshick/yacs
RUN /opt/conda/bin/pip install git+https://github.com/Cadene/pretrained-models.pytorch
RUN export CC=mpicc && export HDF5_MPI="ON" && export HDF5_DIR=/usr/local/hdf5 && /opt/conda/bin/pip install --no-binary=h5py h5py
RUN /opt/conda/bin/conda clean -ya

ENV LD_LIBRARY_PATH .:/usr/local/hdf5/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH /usr/local/hdf5/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
ENV CPATH /usr/local/cuda/include:$CPATH
