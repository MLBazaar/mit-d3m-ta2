#!/bin/bash

function echodo() {
    echo $*
    $*
}

docker build --build-arg UID=$UID -t mit-d3m-ta2 .

COMMANDS=${*:-/bin/bash}
DATASETS=/home/pythia/Projects/d3m/datasets/seed_datasets_current/

echodo docker run -i -t --rm -v $DATASETS:/input -v $(pwd):/home/user -w /home/user -u $UID mit-d3m-ta2 $COMMANDS
