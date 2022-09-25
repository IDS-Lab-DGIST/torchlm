ARG BASE_VERSION=22.07
FROM nvcr.io/nvidia/pytorch:${BASE_VERSION}-py3 as base

ENV DEBIAN_FRONTEND noninteractive


RUN apt-get clean
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get install -y \
                    vim \
                    tmux \ 
                    libgl1-mesa-glx \
                    ffmpeg \
                    libx264-dev \
                    x11-xserver-utils \
                    x11-apps

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

ADD . /root/torchlm
RUN pip install -r /root/torchlm/requirements.txt

RUN addgroup --gid $GROUP_ID $USER_NAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME
USER ${USER_NAME}



# RUN cd /root/torchlm && pip install -e .

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]