FROM python:3.12-slim

WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt
RUN python -m pip install -r /requirements.txt "gradio==4.44.1"

# Download the necessary models when the docker image gets built
RUN mkdir -p /app/models \
    && curl -L -o /app/models/vqgan_imagenet_f16_16384.ckpt "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
