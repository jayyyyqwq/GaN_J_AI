FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyterlab notebook ipywidgets

WORKDIR /tmp
COPY start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

EXPOSE 7860
CMD ["/start_server.sh"]
