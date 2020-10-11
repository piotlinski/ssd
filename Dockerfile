FROM piotrekzie100/pytorch:1.6.0-py38-cuda10.2
RUN apt-get update -yqq && apt-get install -yqq libglib2.0-0 gcc
ADD dist/* ./
RUN pip install *.whl
RUN rm -rf *.whl
ENV MPLCONFIGDIR /tmp/mpl
WORKDIR /app
CMD ["ssd"]
