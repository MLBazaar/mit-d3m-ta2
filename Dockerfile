FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901
# FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10

ARG UID=1000
ARG D3MPORT=45042
WORKDIR /user_dev

# open the grpc listener port
EXPOSE $D3MPORT

RUN mkdir -p /user_dev

# RUN mkdir -p /user_dev && \
#     mkdir -p /user_dev/output && \
#     mkdir -p /user_dev/input && \
#     mkdir -p /user_dev/static && \
RUN ln -s /output /user_dev/output && \
    ln -s /input /user_dev/input && \
    ln -s /static /user_dev/static

# Install requirements
COPY requirements.txt /user_dev/
RUN pip3 install -r /user_dev/requirements.txt

# Copy code
COPY setup.py MANIFEST.in /user_dev/
COPY ta2 /user_dev/ta2
RUN chown -R $UID:$UID /user_dev

# Install project
RUN pip3 install /user_dev
RUN pip3 install ipdb

CMD ["python3", "/user_dev/ta2/ta3/server.py", "-v"]
