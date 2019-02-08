#!/bin/bash

docker build -t mit-d3m-ta2 .

mkdir -p output
chown $USER output

docker run -i -t --rm \
    -p45042:45042 \
    -e D3MTIMEOUT=60 \
    -e D3MINPUTDIR=/input \
    -e D3MOUTPUTDIR=/output \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    -u $UID \
    mit-d3m-ta2
