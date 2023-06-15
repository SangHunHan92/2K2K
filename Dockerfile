FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# ENV DEBIAN_FRONTEND noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt install -y dialog 
RUN apt-get install -y freeglut3-dev libglib2.0-0 libsm6 libxrender1 libxext6 git python3.8 python3.8-dev openexr libopenexr-dev python3-tk libjpeg-dev zlib1g-dev curl python3-distutils python3-apt
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN alias python=python3.8
RUN alias pip=pip3.8
RUN apt install -y libgl1-mesa-dri libegl1-mesa libgbm1 libgl1-mesa-glx libglib2.0-0
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python==4.5.5.64
RUN pip install matplotlib
RUN pip install scipy
RUN pip install pytorch-msssim
RUN pip install trimesh
RUN pip install scikit-image
RUN pip install sklearn
RUN pip install munch
RUN pip install tensorboard
RUN pip install pymeshlab==2022.2
RUN pip install pyembree==0.2.11
RUN apt install python-is-python3
RUN pip install setuptools==59.5.0
RUN pip install pyrender
RUN pip install tqdm