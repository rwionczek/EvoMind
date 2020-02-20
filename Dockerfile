FROM python:3.7

RUN apt-get update && apt-get install -y x11-xserver-utils

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
