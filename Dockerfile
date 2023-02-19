FROM nvidia/cuda:11.6.2-base-ubuntu20.04 as base

RUN apt-get update && apt-get install -y python3-pip python-is-python3 git

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM base as pip-build
WORKDIR /wheels

RUN apt-get update && apt-get -y install git

COPY requirements.txt .
RUN pip install -U pip  \
    && pip wheel -r requirements.txt

FROM base as eval
COPY --from=pip-build /wheels /wheels
WORKDIR /src

ENV TZ=Europe/Berlin
ENV PYTHONPATH=/src/2023-challenge

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6 git && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip install -U pip  \
    && pip install --no-cache-dir \
    --no-index \
    -r /wheels/requirements.txt \
    -f /wheels \
    && rm -rf /wheels

COPY . 2023-challenge/

CMD ["python", "2023-challenge/run.py"]

FROM eval as dev
# For nvidia GPU
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# libgl1-mesa-glx libgl1-mesa-dri for non-nvidia GPU
RUN apt-get update && apt-get -y install xauth tzdata libgl1-mesa-glx libgl1-mesa-dri && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
