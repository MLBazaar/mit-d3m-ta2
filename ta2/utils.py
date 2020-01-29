# -*- coding: utf-8 -*-

import io
import json
import logging
import os
import tarfile
import urllib

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)
DATA_URL = 'https://d3m-data-dai.s3.amazonaws.com/datasets/{}.tar.gz'


def _download(dataset_name, data_path):
    LOGGER.info('Downloading dataset %s into %s folder', dataset_name, data_path)
    url = DATA_URL.format(dataset_name)

    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    LOGGER.debug('Extracting dataset %s into %s folder', dataset_name, data_path)
    with tarfile.open(fileobj=bytes_io, mode='r:gz') as tf:
        tf.extractall(data_path)


def ensure_downloaded(dataset_name, data_path=DATA_PATH):
    if not os.path.exists(data_path):
        LOGGER.debug('Creating data folder %s', data_path)
        os.makedirs(data_path)

    dataset_path = os.path.join(data_path, dataset_name)
    if not os.path.exists(dataset_path):
        _download(dataset_name, data_path)


def dump_pipeline(pipeline, dump_dir, rank=None):
    if not isinstance(pipeline, dict):
        pipeline = pipeline.to_json_structure()

    if 'session' in pipeline:
        pipeline = pipeline.copy()
        del pipeline['session']

    pipeline_filename = pipeline['id'] + '.json'
    pipeline_path = os.path.join(dump_dir, pipeline_filename)
    with open(pipeline_path, 'w') as pipeline_file:
        json.dump(pipeline, pipeline_file, indent=4)

    if rank is not None:
        rank_filename = pipeline['id'] + '.rank'
        rank_path = os.path.join(dump_dir, rank_filename)
        with open(rank_path, 'w') as rank_file:
            print(rank, file=rank_file)


def logging_setup(verbosity=1, logfile=None, logger_name=None, stdout=True):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout or not logfile:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
