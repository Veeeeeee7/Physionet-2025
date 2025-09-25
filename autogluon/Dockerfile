FROM python:3.11-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN pip install --upgrade pip
RUN pip install autogluon sktime mlflow wfdb numba PyWavelets