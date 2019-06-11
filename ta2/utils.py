# -*- coding: utf-8 -*-

import io
import json
import os
import random
import tarfile
import urllib

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)
DATA_URL = 'http://d3m-data-dai.s3.amazonaws.com/{}.tar.gz'


def _download(dataset_name):
    url = DATA_URL.format(dataset_name)

    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    with tarfile.open(fileobj=bytes_io, mode='r:gz') as tf:
        tf.extractall(DATA_PATH)


def ensure_downloaded(dataset_name):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        _download(dataset_name)


def dump_pipeline(pipeline, dump_dir, score=None, rank=None):
    if not isinstance(pipeline, dict):
        pipeline = pipeline.to_json_structure()

    pipeline_filename = pipeline['id'] + '.json'
    pipeline_path = os.path.join(dump_dir, pipeline_filename)
    with open(pipeline_path, 'w') as pipeline_file:
        json.dump(pipeline, pipeline_file, indent=4)

    if score is not None and rank is None:
        rank = (1 - score) + random.random() * 1.e-12   # avoid collisions

    if rank is not None:
        rank_filename = pipeline['id'] + '.rank'
        rank_path = os.path.join(dump_dir, rank_filename)
        with open(rank_path, 'w') as rank_file:
            print(rank, file=rank_file)
