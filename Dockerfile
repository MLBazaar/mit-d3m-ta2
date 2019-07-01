FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190622-073225

ARG D3MPORT=45042
WORKDIR /user_dev

# open the grpc listener port
EXPOSE $D3MPORT

RUN mkdir -p /user_dev && \
    mkdir -p /output && \
    mkdir -p /input

# Install project
COPY setup.py docker_requirements.txt /user_dev/
RUN pip3 install -e /user_dev -r /user_dev/docker_requirements.txt

# Copy code
COPY ta2 /user_dev/ta2

CMD ["python3", "/user_dev/ta2/ta3/server.py", "-v"]
