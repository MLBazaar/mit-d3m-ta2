#!/bin/bash

docker build --build-arg UID=$UID -t mit-d3m-ta2 .

COMMANDS=${*:-/bin/bash}
DATASETS=/home/pythia/Projects/d3m/datasets/seed_datasets_current/

docker run -i -t --rm -v $DATASETS:/input -v $(pwd):/home/user -w /home/user -u $UID mit-d3m-ta2 \
    python3 run_templates.py templates /input/LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA
