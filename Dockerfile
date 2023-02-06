#
FROM --platform=linux/amd64 python:3.9
ENV export DOCKER_DEFAULT_PLATFORM=linux/amd64
# Maintainer info
LABEL maintainer="i60996395@gmail.com"
# New work dir
RUN  mkdir -p  /audio_classification-api

WORKDIR  /audio_classification-api

COPY . .

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc fluidsynth
#
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
