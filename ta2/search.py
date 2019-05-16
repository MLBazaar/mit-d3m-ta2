import glob
import itertools
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from btb.tuning import GP
from d3m.container.dataset import Dataset
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.problem import TaskType
from d3m.runtime import evaluate

from ta2.template import load_template

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINES_DIR = os.path.join(BASE_DIR, 'pipelines')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

TUNING_PARAMETER = 'https://metadata.datadrivendiscovery.org/types/TuningParameter'

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


class StopSearch(Exception):
    pass


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


class PipelineSearcher:

    def _detect_data_modality(self, dataset):
        return 'single_table'

    def _get_task_type_string(self, problem):
        task_type = problem['problem']['task_type']

        if task_type == TaskType.CLASSIFICATION:
            return 'classification'
        elif task_type == TaskType.REGRESSION:
            return 'regression'

        return None

    @staticmethod
    def _find_datasets(input_dir):
        search_path = os.path.join(input_dir, '**', 'datasetDoc.json')
        dataset_docs = glob.glob(search_path, recursive=True)

        datasets = dict()
        for dataset_doc_path in dataset_docs:
            with open(dataset_doc_path, 'r') as dataset_doc_file:
                dataset_doc = json.load(dataset_doc_file)

            dataset_id = dataset_doc['about']['datasetID']
            datasets[dataset_id] = 'file://' + os.path.abspath(dataset_doc_path)

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

    def _get_template(self, dataset, problem):
        task_type = problem['problem']['task_type']

        LOGGER.info("Loading pipeline for task type %s", task_type)

        if task_type == TaskType.CLASSIFICATION:
            return 'gradient_boosting_classification.all_hp.yml'
        elif task_type == TaskType.REGRESSION:
            return 'gradient_boosting_regression.all_hp.yml'

        raise ValueError('Unsupported type of problem')

    def __init__(self, input_dir='input', output_dir='output', dump=True):
        self.input = input_dir
        self.output = output_dir
        self.dump = dump

        self.ranked_dir = os.path.join(self.output, 'pipelines_ranked')
        os.makedirs(self.ranked_dir, exist_ok=True)

        self.datasets = self._find_datasets(input_dir)
        self.data_pipeline = self._load_pipeline('kfold_pipeline.yml')
        self.scoring_pipeline = self._load_pipeline('scoring_pipeline.yml')

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
        )

        pipeline.cv_scores = [score.value[0] for score in all_scores]
        pipeline.score = np.mean(pipeline.cv_scores)

        return pipeline.score

    def _save_pipeline(self, pipeline, normalized_score):
        pipeline_json = pipeline.to_json_structure()
        pipeline_json['score'] = pipeline.score
        self.solutions.append(pipeline_json)

        if self.dump:
            rank = (1 - normalized_score) + random.random() * 1.e-12   # to avoid collisions
            pipeline_json['pipeline_rank'] = rank
            # pipeline_json['pipeline_rank'] = pipeline.rank

            pipeline_filename = pipeline.id + '.json'
            pipeline_path = os.path.join(self.ranked_dir, pipeline_filename)

            with open(pipeline_path, 'w') as pipeline_file:
                json.dump(pipeline_json, pipeline_file, indent=4)

    @staticmethod
    def _new_pipeline(pipeline, hyperparams=None):
        hyperparams = to_dicts(hyperparams) if hyperparams else dict()

        new_pipeline = Pipeline(context=Context.TESTING)
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
            raise StopSearch()

    def stop(self):
        self._stop = True

    def setup_search(self, timeout):
        self.solutions = list()
        self._stop = False
        self.done = False

        self.start_time = datetime.now()
        self.timeout = timeout
        self.max_end_time = None
        if self.timeout:
            self.max_end_time = self.start_time + timedelta(seconds=self.timeout)

        LOGGER.info("Timeout: %s; Max end: %s", self.timeout, self.max_end_time)

    def search(self, problem, timeout=None, budget=None):

        self.setup_search(timeout)

        dataset_id = problem['inputs'][0]['dataset_id']
        dataset = Dataset.load(self.datasets[dataset_id])
        metric = problem['problem']['performance_metrics'][0]['metric']

        LOGGER.info("Loading the template and the tuner")
        template_name = self._get_template(dataset, problem)
        template, tunables, defaults = load_template(template_name)
        tuner = GP(tunables, r_minimum=10)

        best_pipeline = None
        best_score = None
        best_normalized = 0

        if budget is not None:
            iterator = range(budget)
        else:
            iterator = itertools.count()   # infinite range

        try:
            proposal = defaults
            for i in iterator:
                self.check_stop()
                pipeline = self._new_pipeline(template, proposal)

                params = '\n'.join('{}: {}'.format(k, v) for k, v in proposal.items())
                LOGGER.info("Scoring pipeline %s: %s\n%s", i + 1, pipeline.id, params)
                try:
                    score = self.score_pipeline(dataset, problem, pipeline)
                    normalized_score = metric.normalize(score)
                except Exception:
                    LOGGER.exception("Error scoring pipeline %s", pipeline.id)
                    score = None
                    normalized_score = 0.0

                try:
                    self._save_pipeline(pipeline, normalized_score)
                except Exception:
                    LOGGER.exception("Error saving pipeline %s", pipeline.id)

                tuner.add(proposal, normalized_score)
                LOGGER.info("Pipeline %s score: %s - %s", pipeline.id, score, normalized_score)

                if normalized_score > best_normalized:
                    LOGGER.info("New best pipeline found! %s > %s", score, best_score)
                    best_pipeline = pipeline.id
                    best_score = score
                    best_normalized = normalized_score

                proposal = tuner.propose(1)

        except StopSearch:
            pass

        self.done = True
        return {
            'pipeline': best_pipeline,
            'score': best_score,
            'template': template_name,
            'data_modality': self._detect_data_modality(dataset),
            'task_type': self._get_task_type_string(problem)
        }
