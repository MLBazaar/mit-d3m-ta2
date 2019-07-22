FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190622-073225

ARG D3MPORT=45042
WORKDIR /user_dev

# open the grpc listener port
EXPOSE $D3MPORT

RUN mkdir -p /user_dev && \
    mkdir -p /user_dev/output && \
    mkdir -p /user_dev/input && \
    mkdir -p /user_dev/static && \
    ln -s /user_dev/output /output && \
    ln -s /user_dev/input /input && \
    ln -s /user_dev/static /static

# Install requirements
COPY docker_requirements.txt /user_dev/
RUN pip3 install -r /user_dev/docker_requirements.txt

# Copy code
COPY setup.py MANIFEST.in /user_dev/
COPY ta2 /user_dev/ta2

# Install project
RUN pip3 install /user_dev

CMD ["python3", "/user_dev/ta2/ta3/server.py", "-v"]
