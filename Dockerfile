# using this since pytorch images still use python 3.7
FROM anibali/pytorch:1.5.0-cuda10.2
USER root
ARG PYPI_USERNAME=trasee_rd
ARG PYPI_PASSWORD
ARG SSD_VERSION=0.1.0+47e45e4
RUN apt-get update -yqq && apt-get install -yqq libglib2.0-0
RUN pip install -i https://${PYPI_USERNAME}:${PYPI_PASSWORD}@pypi.trasee.io/simple/ ssd==${SSD_VERSION}
VOLUME /app
VOLUME /app/data
VOLUME /app/models
WORKDIR /app
ENTRYPOINT ["ssd"]
