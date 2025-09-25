FROM python:3.9-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN pip install pip==24.0
RUN pip install --no-cache-dir -r requirements.txt

## Include the following line if you have a requirements.txt file.
RUN mkdir -p /challenge/model/ckpts && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='wanglab/ecg-fm', filename='mimic_iv_ecg_physionet_pretrained.pt', local_dir='/challenge/ckpts')"
