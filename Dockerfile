FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

COPY ./requirements.txt /requirements.txt
RUN python -m pip install -r /requirements.txt gradio

# Download the necessary models when the docker image gets built
RUN mkdir -p /app/models \
    && curl -L -o /app/models/vqgan_imagenet_f16_16384.ckpt "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
