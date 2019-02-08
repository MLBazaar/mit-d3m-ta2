FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2019.1.21

ARG D3MPORT=45042

RUN mkdir -p /user_dev && \
    mkdir -p /output && \
    mkdir -p /input

# RUN pip3 install --upgrade pip

# open the grpc listener port
EXPOSE $D3MPORT

# copy code
COPY requirements.txt setup.py /user_dev/
COPY ta2 /user_dev/ta2

RUN pip3 install -e /user_dev -r /user_dev/requirements.txt

WORKDIR /user_dev
ENTRYPOINT ["python3", "/user_dev/ta2/ta3/server.py", "-v"]
