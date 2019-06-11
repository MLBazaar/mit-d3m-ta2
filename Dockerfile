FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190611-060317

ARG D3MPORT=45042

# open the grpc listener port
EXPOSE $D3MPORT

RUN mkdir -p /user_dev && \
    mkdir -p /output && \
    mkdir -p /input

COPY requirements.txt /user_dev/
RUN pip3 install -r /user_dev/requirements.txt

# Install project
COPY setup.py /user_dev/
RUN pip3 install -e /user_dev

# Copy code
COPY ta2 /user_dev/ta2
COPY ta2_test.py /user_dev/

WORKDIR /user_dev
CMD ["python3", "/user_dev/ta2/ta3/server.py", "-v"]
