FROM python:3.8

WORKDIR /app

ADD ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

ADD . /app
