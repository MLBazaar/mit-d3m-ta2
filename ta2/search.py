import glob
import itertools
import json
import logging
import os
import random
import signal
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from btb.tuning import GP
from d3m.container.dataset import Dataset
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.problem import TaskType
from d3m.runtime import DEFAULT_SCORING_PIPELINE_PATH, evaluate

from ta2.template import load_template
from ta2.utils import dump_pipeline

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINES_DIR = os.path.join(BASE_DIR, 'pipelines')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

TUNING_PARAMETER = 'https://metadata.datadrivendiscovery.org/types/TuningParameter'

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Templates(Enum):
    # SINGLE_TABLE_CLASSIFICATION = 'gradient_boosting_classification.hp.yml'
    # SINGLE_TABLE_CLASSIFICATION = 'xgb_classification.hp.yml'
    SINGLE_TABLE_CLASSIFICATION = 'single_table_classification_encoding_xgb.yml'
    # SINGLE_TABLE_REGRESSION = 'gradient_boosting_regression.hp.yml'
    SINGLE_TABLE_REGRESSION = 'single_table_regression_xgb.yml'
    MULTI_TABLE_CLASSIFICATION = 'multi_table_classification_dfs_xgb.yml'
    MULTI_TABLE_REGRESSION = 'multi_table_regression_dfs_xgb.yml'
    # TIMESERIES_CLASSIFICATION = 'time_series_xgb_classification.yml'
    TIMESERIES = 'time_series_classification_k_neighbors_kn.yml'
    IMAGE_REGRESSION = 'image_regression_resnet50_xgb.yml'
    IMAGE_CLASSIFICATION = 'image_classification_resnet50_xgb.yml'
    IMAGE_OBJECT_DETECTION = 'image_object_detection_yolo.yml'
    TEXT_CLASSIFICATION = 'text_classification_encoding_xgb.yml'
    TEXT_REGRESSION = 'text_regression_encoding_xgb.yml'
    GRAPH_COMMUNITY_DETECTION = 'graph_community_detection.yml'
    GRAPH_LINK_PREDICTION = 'graph_link_prediction_distil.yml'
    GRAPH_MATCHING = 'graph_matching_jhu.yml'
    SINGLE_TABLE_SEMI_CLASSIFICATION = 'single_table_semi_classification_autonbox.yml'


def detect_data_modality(dataset_doc_path):
    with open(dataset_doc_path) as f:
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
    task_type = problem['problem']['task_type'].name.lower()
    task_subtype = problem['problem']['task_subtype'].name.lower()

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


FILE_COLLECTION = 'https://metadata.datadrivendiscovery.org/types/FilesCollection'
GRAPH = 'https://metadata.datadrivendiscovery.org/types/Graph'
EDGE_LIST = 'https://metadata.datadrivendiscovery.org/types/EdgeList'


class PipelineSearcher:

    @staticmethod
    def _find_datasets(input_dir):
        search_path = os.path.join(input_dir, '**', 'datasetDoc.json')
        dataset_docs = glob.glob(search_path, recursive=True)

        datasets = dict()
        for dataset_doc_path in dataset_docs:
            with open(dataset_doc_path, 'r') as dataset_doc_file:
                dataset_doc = json.load(dataset_doc_file)

            dataset_id = dataset_doc['about']['datasetID']
            dataset_name = os.path.basename(
                os.path.realpath(os.path.join(dataset_doc_path, *[os.pardir] * 3))
            )
            datasets[dataset_id] = dataset_name, 'file://' + os.path.abspath(dataset_doc_path)

        return datasets

    def _load_pipeline(self, pipeline):
        if pipeline.endswith('.yml'):
            loader = Pipeline.from_yaml
        else:
            loader = Pipeline.from_json
            if not pipeline.endswith('.json'):
                pipeline += '.json'

        path = os.path.join(PIPELINES_DIR, pipeline)
        with open(path, 'r') as pipeline_file:
            return loader(string_or_file=pipeline_file)

    def _get_template(self, data_modality, task_type):
        LOGGER.info("Loading template for data modality %s and task type %s",
                    data_modality, task_type)

        template = None
        if data_modality == 'single_table':
            if task_type == TaskType.CLASSIFICATION.name.lower():
                template = Templates.SINGLE_TABLE_CLASSIFICATION
            elif task_type == TaskType.REGRESSION.name.lower():
                template = Templates.SINGLE_TABLE_REGRESSION
            elif task_type == TaskType.COLLABORATIVE_FILTERING.name.lower():
                template = Templates.SINGLE_TABLE_REGRESSION
            elif task_type == TaskType.TIME_SERIES_FORECASTING.name.lower():
                template = Templates.SINGLE_TABLE_REGRESSION
            elif task_type == TaskType.SEMISUPERVISED_CLASSIFICATION.name.lower():
                template = Templates.SINGLE_TABLE_SEMI_CLASSIFICATION
        if data_modality == 'multi_table':
            if task_type == TaskType.CLASSIFICATION.name.lower():
                template = Templates.MULTI_TABLE_CLASSIFICATION
            elif task_type == TaskType.REGRESSION.name.lower():
                template = Templates.MULTI_TABLE_REGRESSION
        elif data_modality == 'text':
            if task_type == TaskType.CLASSIFICATION.name.lower():
                template = Templates.TEXT_CLASSIFICATION
            elif task_type == TaskType.REGRESSION.name.lower():
                template = Templates.TEXT_REGRESSION
        if data_modality == 'timeseries':
            template = Templates.TIMESERIES
            # if task_type == TaskType.CLASSIFICATION.name.lower():
            #     template = Templates.TIMESERIES_CLASSIFICATION
            # elif task_type == TaskType.REGRESSION.name.lower():
            #     template = Templates.TIMESERIES_REGRESSION
        elif data_modality == 'image':
            if task_type == TaskType.CLASSIFICATION.name.lower():
                template = Templates.IMAGE_CLASSIFICATION
            elif task_type == TaskType.REGRESSION.name.lower():
                template = Templates.IMAGE_REGRESSION
            elif task_type == TaskType.OBJECT_DETECTION.name.lower():
                template = Templates.IMAGE_OBJECT_DETECTION
        if data_modality == 'graph':
            if task_type == TaskType.COMMUNITY_DETECTION.name.lower():
                template = Templates.GRAPH_COMMUNITY_DETECTION
            elif task_type == TaskType.LINK_PREDICTION.name.lower():
                template = Templates.GRAPH_LINK_PREDICTION
            elif task_type == TaskType.GRAPH_MATCHING.name.lower():
                template = Templates.GRAPH_MATCHING
            elif task_type == TaskType.VERTEX_CLASSIFICATION.name.lower():
                template = Templates.SINGLE_TABLE_CLASSIFICATION

        if template:
            return template.value

        return Templates.SINGLE_TABLE_CLASSIFICATION.value
        # raise ValueError('Unsupported problem')

    def __init__(self, input_dir='input', output_dir='output', static_dir='static',
                 dump=False, hard_timeout=False):
        self.input = input_dir
        self.output = output_dir
        self.static = static_dir
        self.dump = dump
        self.hard_timeout = hard_timeout

        self.ranked_dir = os.path.join(self.output, 'pipelines_ranked')
        self.scored_dir = os.path.join(self.output, 'pipelines_scored')
        self.searched_dir = os.path.join(self.output, 'pipelines_searched')
        os.makedirs(self.ranked_dir, exist_ok=True)
        os.makedirs(self.scored_dir, exist_ok=True)
        os.makedirs(self.searched_dir, exist_ok=True)

        self.solutions = list()
        self.datasets = self._find_datasets(input_dir)
        self.data_pipeline = self._load_pipeline('kfold_pipeline.yml')
        self.scoring_pipeline = self._load_pipeline(DEFAULT_SCORING_PIPELINE_PATH)

    def score_pipeline(self, dataset, problem, pipeline, metrics=None, random_seed=0,
                       folds=5, stratified=False, shuffle=False):
        problem_metrics = problem['problem']['performance_metrics']
        metrics = metrics or problem_metrics
        data_params = {
            'number_of_folds': json.dumps(folds),
            'stratified': json.dumps(stratified),
            'shuffle': json.dumps(shuffle),
        }
        all_scores, all_results = evaluate(
            pipeline,
            self.data_pipeline,
            self.scoring_pipeline,
            problem,
            [dataset],
            data_params,
            metrics,
            context=Context.TESTING,
            random_seed=random_seed,
            data_random_seed=random_seed,
            scoring_random_seed=random_seed,
            volumes_dir=self.static,
        )

        if not all_scores:
            failed_result = all_results[-1]
            message = failed_result.pipeline_run.status['message']
            LOGGER.error(message)
            raise failed_result.error.__cause__

        pipeline.cv_scores = [score.value[0] for score in all_scores]
        pipeline.score = np.mean(pipeline.cv_scores)

    def _save_pipeline(self, pipeline):
        pipeline_dict = pipeline.to_json_structure()

        if pipeline.score is None:
            dump_pipeline(pipeline_dict, self.searched_dir)
        else:
            dump_pipeline(pipeline_dict, self.scored_dir)

            rank = (1 - pipeline.normalized_score) + random.random() * 1.e-12   # avoid collisions
            if self.dump:
                dump_pipeline(pipeline_dict, self.ranked_dir, rank)

            pipeline_dict['rank'] = rank
            pipeline_dict['score'] = pipeline.score
            pipeline_dict['normalized_score'] = pipeline.normalized_score
            self.solutions.append(pipeline_dict)

    @staticmethod
    def _new_pipeline(pipeline, hyperparams=None):
        hyperparams = to_dicts(hyperparams) if hyperparams else dict()

        new_pipeline = Pipeline()
        for input_ in pipeline.inputs:
            new_pipeline.add_input(name=input_['name'])

        for step_id, old_step in enumerate(pipeline.steps):
            new_step = PrimitiveStep(primitive=old_step.primitive)
            for name, argument in old_step.arguments.items():
                new_step.add_argument(
                    name=name,
                    argument_type=argument['type'],
                    data_reference=argument['data']
                )
            for output in old_step.outputs:
                new_step.add_output(output)

            new_hyperparams = hyperparams.get(str(step_id), dict())
            for name, hyperparam in old_step.hyperparams.items():
                if name not in new_hyperparams:
                    new_step.add_hyperparameter(
                        name=name,
                        argument_type=ArgumentType.VALUE,
                        data=hyperparam['data']
                    )

            for name, value in new_hyperparams.items():
                new_step.add_hyperparameter(
                    name=name,
                    argument_type=ArgumentType.VALUE,
                    data=value
                )

            new_pipeline.add_step(new_step)

        for output in pipeline.outputs:
            new_pipeline.add_output(
                name=output['name'],
                data_reference=output['data']
            )

        new_pipeline.cv_scores = list()
        new_pipeline.score = None

        return new_pipeline

    def check_stop(self):
        now = datetime.now()

        if (self._stop or (self.timeout and (now > self.max_end_time))):
            raise KeyboardInterrupt()

    def stop(self):
        self._stop = True

    def _timeout(self, *args, **kwargs):
        raise KeyboardInterrupt()

    def setup_search(self):
        self.solutions = list()
        self._stop = False
        self.done = False

        self.start_time = datetime.now()
        self.max_end_time = None
        if self.timeout:
            self.max_end_time = self.start_time + timedelta(seconds=self.timeout)

            if self.hard_timeout:
                signal.signal(signal.SIGALRM, self._timeout)
                signal.alarm(self.timeout)

        LOGGER.info("Timeout: %s (Hard: %s); Max end: %s",
                    self.timeout, self.hard_timeout, self.max_end_time)

    def search(self, problem, timeout=None, budget=None, template_name=None):

        self.timeout = timeout
        best_pipeline = None
        best_score = None
        best_normalized = 0
        data_modality = None
        task_type = None
        task_subtype = None
        iteration = 0
        error = None

        try:
            dataset_id = problem['inputs'][0]['dataset_id']
            dataset_name, dataset_path = self.datasets[dataset_id]
            dataset = Dataset.load(dataset_path)
            metric = problem['problem']['performance_metrics'][0]['metric']

            data_modality = detect_data_modality(dataset_path[7:])
            task_type = problem['problem']['task_type'].name.lower()
            task_subtype = problem['problem']['task_subtype'].name.lower()

            LOGGER.info("Searching dataset %s: %s/%s/%s",
                        dataset_name, data_modality, task_type, task_subtype)
            LOGGER.info("Loading the template and the tuner")
            if template_name is None:
                template_name = self._get_template(data_modality, task_type)

            template, tunables, defaults = load_template(template_name)
            tuner = GP(tunables, r_minimum=10)

            if budget is not None:
                iterator = range(budget)
            else:
                iterator = itertools.count()   # infinite range

            self.setup_search()

            first = True
            proposal = defaults
            for iteration in iterator:
                self.check_stop()
                pipeline = self._new_pipeline(template, proposal)

                params = '\n'.join('{}: {}'.format(k, v) for k, v in proposal.items())
                LOGGER.info("Scoring pipeline %s: %s\n%s", iteration + 1, pipeline.id, params)
                try:
                    self.score_pipeline(dataset, problem, pipeline)
                    pipeline.normalized_score = metric.normalize(pipeline.score)
                except Exception:
                    if first:
                        # We want this to be reported
                        raise

                    LOGGER.exception("Error scoring pipeline %s", pipeline.id)
                    pipeline.score = None
                    pipeline.normalized_score = 0.0

                try:
                    self._save_pipeline(pipeline)
                except Exception:
                    LOGGER.exception("Error saving pipeline %s", pipeline.id)

                tuner.add(proposal, pipeline.normalized_score)
                LOGGER.info("Pipeline %s score: %s - %s",
                            pipeline.id, pipeline.score, pipeline.normalized_score)

                if pipeline.normalized_score > best_normalized:
                    LOGGER.info("New best pipeline found! %s is better than %s",
                                pipeline.score, best_score)
                    best_pipeline = pipeline.id
                    best_score = pipeline.score
                    best_normalized = pipeline.normalized_score

                proposal = tuner.propose(1)
                first = False

        except KeyboardInterrupt:
            pass
        except Exception as ex:
            LOGGER.exception("Error processing dataset %s", dataset)
            error = '{}: {}'.format(type(ex).__name__, ex)

        finally:
            if self.timeout and self.hard_timeout:
                signal.alarm(0)

        self.done = True
        return {
            'pipeline': best_pipeline,
            'cv_score': best_score,
            'template': template_name,
            'data_modality': data_modality,
            'task_type': task_type,
            'task_subtype': task_subtype,
            'tuning_iterations': iteration,
            'error': error
        }
