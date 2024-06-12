FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# set it to 1 to pre-compile deepspeed during bulding,
# otherwise, deepspeed will be installed in JIT-mode
ENV DS_BUILD_OPS=0

RUN mkdir ./workspace
WORKDIR ./workspace

RUN apt-get update && apt-get -y upgrade
RUN apt-get install git
RUN apt-get install -y python3.10 python3-pip git

# mapping python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# installing dependencies 
RUN pip install --no-cache-dir \
    huggingface_hub \
    ruff \
    pre-commit \
    gymnasium \ 
    minigrid \
    matplotlib \ 
    wandb \
    numpy \
    rich \
    h5py \
    tqdm \
    imageio \
    imageio-ffmpeg \
    pyrallis

RUN pip install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic \
    torch \
    flash-attn \
    deepspeed

RUN pip install "jax[cuda12]"
RUN pip install --no-cache-dir \
    xminigrid \
    flax \
    chex \
    distrax \
    optax \
    orbax-checkpoint \
