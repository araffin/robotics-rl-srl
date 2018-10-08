FROM nvidia/cuda:9.0-base-ubuntu16.04

#  $ docker build . -t continuumio/miniconda3:latest -t continuumio/miniconda3:4.5.4
#  $ docker run --rm -it continuumio/miniconda3:latest /bin/bash
#  $ docker push continuumio/miniconda3:latest
#  $ docker push continuumio/miniconda3:4.5.4

ENV CODE_DIR /root/code

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git

RUN apt-get -y install libopenmpi-dev zlib1g-dev cmake libglib2.0-0\
  libsm6 libxext6 libfontconfig1 libxrender1 swig build-essential libgtk2.0-0\
  libgl1-mesa-dev libglu1-mesa-dev x11-xserver-utils

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

COPY environment.yml /tmp/
RUN conda env create --file /tmp/environment.yml
RUN conda clean -ya

RUN echo "conda activate py35" >> ~/.bashrc


ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
