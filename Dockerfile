FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN sudo apt-get install gcc python3-dev
RUN pip install --no-cache-dir -r requirements.txt
