FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir jupyterlab notebook ipywidgets

WORKDIR /tmp
COPY start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

EXPOSE 7860
CMD ["/start_server.sh"]
