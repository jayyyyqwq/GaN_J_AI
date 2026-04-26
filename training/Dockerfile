FROM python:3.10-slim

RUN apt-get update && apt-get install -y git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyterlab notebook ipywidgets

WORKDIR /tmp
COPY start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

EXPOSE 7860
CMD ["/start_server.sh"]
