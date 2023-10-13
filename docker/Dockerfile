FROM ubuntu:16.04
ARG DEBIAN_FRONTEND=noninteractive

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#     wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
#     /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
#     rm /Anaconda2-2.5.0-Linux-x86_64.sh


RUN sed -i~orig -e 's/# deb-src/deb-src/' /etc/apt/sources.list && apt-get update --fix-missing
RUN apt-get install -y curl grep sed dpkg && \
    curl -L "https://github.com/krallin/tini/releases/download/v0.19.0/tini_0.19.0.deb" > tini.deb
RUN dpkg -i tini.deb
RUN rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
# ENV LANG C.UTF-8
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Special Deps ==========
RUN apt-get update
RUN apt-get -y install git make cmake unzip
RUN python -m pip install --upgrade "pip < 21.0"
RUN pip install --upgrade pip
RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
# MUJOCO requires graphics stuff (Why?)
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN apt-get install -y vim ack-grep
# usual pip install pygame will fail
RUN apt-get build-dep -y python-pygame
RUN pip install Pillow
RUN apt install -y libopenmpi-dev

# ========== OpenAI Gym ==========
RUN apt-get -y install libgtk2.0-0
#RUN apt-get -y install ffmpeg
RUN apt-get -y install libav-tools
CMD alias ffmpeg="avconv"

# ========== Setup mujoco ========
CMD mkdir /root/.mujoco
RUN wget https://www.roboti.us/download/mjpro131_linux.zip
RUN unzip mjpro131_linux.zip -d /root/.mujoco
RUN wget -P /root/.mujoco https://www.roboti.us/file/mjkey.txt
#ADD mjkey.txt /root/.mujoco/mjkey.txt
RUN apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev

# ========= Cassie Environment ===========
RUN apt-get install -y llvm-5.0
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y clang-5.0

# ========== Add codebase stub ==========
CMD mkdir -p /root/code
ADD environment.yml /root/code/environment.yml
RUN conda env create -f /root/code/environment.yml

ENV PYTHONPATH /root/code/learning_to_adapt:$PYTHONPATH
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro131/bin:$LD_LIBRARY_PATH
ENV PATH /opt/conda/envs/learning_to_adapt/bin:$PATH
RUN echo "source activate learning_to_adapt" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code
RUN git clone https://github.com/iclavera/learning_to_adapt.git


