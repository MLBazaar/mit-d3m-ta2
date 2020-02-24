# -*- coding: utf-8 -*-

import json
import logging
import os
from collections import defaultdict

import numpy as np
from d3m.container.dataset import Dataset
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import Problem

LOGGER = logging.getLogger(__name__)


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


def box_log(message, strong=False):
    char = '#' if strong else '*'
    line = char * max(len(line) for line in message.split('\n'))
    LOGGER.warn('\n'.join(('', line, message, line)))


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


def get_dataset_name(problem):
    dataset_name = problem['inputs'][0]['dataset_id']
    if dataset_name.endswith('_dataset'):
        dataset_name = dataset_name[:-len('_dataset')]

    return dataset_name


def detect_data_modality(dataset):
    dataset_doc_path = dataset.metadata.query(())['location_uris'][0]
    with open(dataset_doc_path[7:]) as f:
        dataset_doc = json.load(f)

    resources = list()
    for resource in dataset_doc['dataResources']:
        resources.append(resource['resType'])

    if len(resources) == 1:
        return 'single_table'
    else:
        for resource in resources:
            if resource == 'edgeList':
                return 'graph'
            elif resource not in ('table', 'raw'):
                return resource

    return 'multi_table'


def get_dataset_details(dataset, problem):
    data_modality = detect_data_modality(dataset)
    task_type = problem['problem']['task_keywords'][0].name.lower()
    task_subtype = problem['problem']['task_keywords'][1].name.lower()

    return data_modality, task_type, task_subtype


def to_dicts(hyperparameters):

    params_tree = defaultdict(dict)
    for (block, hyperparameter), value in hyperparameters.items():
        if isinstance(value, np.integer):
            value = int(value)

        elif isinstance(value, np.floating):
            value = float(value)

        elif isinstance(value, np.ndarray):
            value = value.tolist()

        elif isinstance(value, np.bool_):
            value = bool(value)

        elif value == 'None':
            value = None

        params_tree[block][hyperparameter] = value

    return params_tree


def load_dataset(root_path, phase, inner_phase=None):
    inner_phase = inner_phase or phase
    path = os.path.join(root_path, phase, 'dataset_' + inner_phase, 'datasetDoc.json')
    if os.path.exists(path):
        return Dataset.load(dataset_uri='file://' + os.path.abspath(path))
    else:
        path = os.path.join(root_path, phase, 'dataset_' + phase, 'datasetDoc.json')
        return Dataset.load(dataset_uri=path)


def load_problem(root_path, phase):
    path = os.path.join(root_path, phase, 'problem_' + phase, 'problemDoc.json')
    return Problem.load(problem_uri=path)


def load_pipeline(pipeline_path):
    with open(pipeline_path, 'r') as pipeline_file:
        if pipeline_path.endswith('.json'):
            return Pipeline.from_json(pipeline_file)
        else:
            return Pipeline.from_yaml(pipeline_file)


def get_datasets(input_dir, datasets=None, data_modality=None, task_type=None):
    if not datasets:
        datasets = os.listdir(input_dir)

    for dataset_name in datasets:
        dataset_root = os.path.join(input_dir, dataset_name)
        if not os.path.exists(dataset_root):
            dataset_root += '_MIN_METADATA'

        dataset_root = 'file://' + os.path.abspath(dataset_root)

        try:
            dataset = load_dataset(dataset_root, 'TRAIN')
            problem = load_problem(dataset_root, 'TRAIN')
            data_modality, task_type = get_dataset_details(dataset, problem)[:2]
        except Exception:
            continue

        if data_modality and not data_modality == data_modality:
            continue
        if task_type and not task_type == task_type:
            continue

        dataset.name = dataset_name.replace('_MIN_METADATA', '')
        yield dataset, problem, data_modality, task_type
