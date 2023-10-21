# ========== Anaconda ==========
# uses Anaconda2:2.5.0
FROM kentwait/anaconda-mpi


RUN sudo apt-get update
RUN sudo apt-get -y install git make cmake unzip
RUN pip install awscli
# ALE requires zlib
RUN sudo apt-get -y install zlib1g-dev
# MUJOCO requires graphics stuff (Why?)
RUN sudo apt-get -y build-dep glfw
RUN sudo apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN sudo apt-get install -y vim ack-grep
# RUN pip install --upgrade pip
# usual pip install pygame will fail
RUN sudo apt-get build-dep -y python-pygame
RUN #sudo apt-get update
RUN pip install Pillow
RUN sudo apt install -y libopenmpi-dev

# ========== OpenAI Gym ==========
RUN sudo apt-get -y install libgtk2.0-0
#RUN apt-get -y install ffmpeg
RUN sudo apt-get -y install libav-tools
CMD alias ffmpeg="avconv"

# ========== Setup mujoco ========
CMD mkdir /root/.mujoco
RUN wget https://www.roboti.us/download/mjpro131_linux.zip
RUN sudo unzip mjpro131_linux.zip -d /root/.mujoco
#ADD mjkey.txt /root/.mujoco/mjkey.txt
RUN sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev

# ========= Cassie Environment ===========
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
RUN sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main"
RUN echo "deb http://ppa.launchpad.net/keithw/glfw3/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/fillwave_ext.list
RUN echo "deb-src http://ppa.launchpad.net/keithw/glfw3/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/fillwave_ext.list
RUN sudo apt-get update
RUN sudo apt-get install -y llvm
RUN sudo apt-get install --force-yes -y libglfw3-dev
RUN sudo apt-get install -y clang

#RUN sudo apt-get install -y llvm-5.0
#RUN sudo apt-get install -y libglfw3-dev
#RUN sudo apt-get install -y clang-5.0

# ========== Add codebase stub ==========
CMD mkdir /root/code
ADD environment.yml /root/code/environment.yml
WORKDIR /root/code
RUN conda env create -f environment.yml

USER root
WORKDIR /
ENV PYTHONPATH /root/code/learning_to_adapt:$PYTHONPATH
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro131/bin:$LD_LIBRARY_PATH
ENV PATH /opt/conda/envs/learning_to_adapt/bin:$PATH
RUN sudo echo "source activate learning_to_adapt" >> ~/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

RUN pip install --upgrade pip
RUN #pip install tensorflow==1.13.1
