# -*- coding: utf-8 -*-

import io
import os
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
