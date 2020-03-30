FROM python:3.6

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r ./requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/"

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/