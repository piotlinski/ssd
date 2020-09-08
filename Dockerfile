# using this since pytorch images still use python 3.7
FROM anibali/pytorch:1.5.0-cuda10.2
USER root
ARG UID=1000
ARG GID=1000
RUN apt-get update -yqq && apt-get install -yqq libglib2.0-0 gcc
ADD dist/* ./
RUN pip install *.whl
RUN rm -rf *.whl
USER ${UID}:${GID}
WORKDIR /app
ENTRYPOINT ["pyssd"]
