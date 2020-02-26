FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9-20200212-063959

ARG UID=1000
ARG D3MPORT=45042
WORKDIR /user_dev

# open the grpc listener port
EXPOSE $D3MPORT

RUN mkdir -p /user_dev

RUN ln -s /output /user_dev/output && \
    ln -s /input /user_dev/input && \
    ln -s /static /user_dev/static

# Copy code
COPY setup.py MANIFEST.in /user_dev/
RUN pip3 install -e /user_dev ipdb

COPY ta2 /user_dev/ta2
RUN mkdir -p /.local && chmod 777 /.local

CMD ["python3", "/user_dev/ta2/ta3/server.py", "-v"]
