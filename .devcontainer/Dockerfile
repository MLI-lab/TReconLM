FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update && apt-get install -y git screen build-essential \
    cm-super texlive-fonts-recommended texlive-full

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace
SHELL ["/bin/bash", "--login", "-c"]

CMD ["/bin/bash"]