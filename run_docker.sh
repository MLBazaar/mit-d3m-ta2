#!/bin/bash

# docker build -t mit-d3m-ta2 .

rm -r output
mkdir -p output
chown $USER output

docker run -i -t --rm \
    -p45042:45042 \
    -e D3MTIMEOUT=60 \
    -e D3MINPUTDIR=/input \
    -e D3MOUTPUTDIR=/output \
    -e D3MSTATICDIR=/static \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    -v $(pwd)/static:/static \
    -u $UID \
    mit-d3m-ta2
