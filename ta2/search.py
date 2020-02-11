import json
import logging
import os
import random
import signal
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Manager, Process

import numpy as np
import pandas as pd
from btb.session import BTBSession
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.runtime import DEFAULT_SCORING_PIPELINE_PATH
from d3m.runtime import evaluate as d3m_evaluate
from datamart import DatamartQuery
from datamart_rest import RESTDatamart

from ta2.loader import LazyLoader
from ta2.utils import dump_pipeline

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINES_DIR = os.path.join(BASE_DIR, 'pipelines')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

DATAMART_URL = os.getenv('DATAMART_URL_NYU', 'https://datamart.d3m.vida-nyu.org')

SUBPROCESS_PRIMITIVES = [
    'd3m.primitives.natural_language_processing.lda.Fastlvm',
    'd3m.primitives.feature_construction.sdne.DSBOX',
    'd3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec',
]

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


class PipelineSearcher:

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

    def _get_templates(self, dataset_name, data_modality, task_type):
        LOGGER.info('Loading template for dataset %s', dataset_name)

        df = pd.read_csv(os.path.join(TEMPLATES_DIR, 'templates_with_z_score.csv'))

        templates = df[df['name'] == dataset_name].sort_values('z_score', ascending=False)

        problem_type = '{}_{}'.format(data_modality, task_type)
        df['match'] = df['name'] == dataset_name
        templates = df[df['problem_type'] == problem_type]
        templates = templates.sort_values(['match', 'z_score'], ascending=False)
        templates = templates.drop_duplicates(subset=['pipeline_id'], keep='first')

        self.found_by_name = df['match'].any()

        return templates.pipeline_id.values

    def __init__(self, input_dir='input', output_dir='output', static_dir='static',
                 dump=False, hard_timeout=False, ignore_errors=False):
        self.input = input_dir
        self.output = output_dir
        self.static = static_dir
        self.dump = dump
        self.hard_timeout = hard_timeout
        self.subprocess = None

        self.runs_dir = os.path.join(self.output, 'pipeline_runs')
        self.ranked_dir = os.path.join(self.output, 'pipelines_ranked')
        self.scored_dir = os.path.join(self.output, 'pipelines_scored')
        self.searched_dir = os.path.join(self.output, 'pipelines_searched')
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.ranked_dir, exist_ok=True)
        os.makedirs(self.scored_dir, exist_ok=True)
        os.makedirs(self.searched_dir, exist_ok=True)

        self.solutions = list()
        self.data_pipeline = self._load_pipeline('kfold_pipeline.yml')
        self.scoring_pipeline = self._load_pipeline(DEFAULT_SCORING_PIPELINE_PATH)
        self.ignore_errors = ignore_errors

    @staticmethod
    def _evaluate(out, pipeline, *args, **kwargs):
        LOGGER.info('Running d3m.runtime.evalute on pipeline %s', pipeline.id)
        results = d3m_evaluate(pipeline, *args, **kwargs)

        LOGGER.info('Returning results for %s', pipeline.id)
        out.extend(results)

    def subprocess_evaluate(self, pipeline, *args, **kwargs):
        LOGGER.info('Evaluating pipeline %s in a subprocess', pipeline.id)
        with Manager() as manager:
            output = manager.list()
            process = Process(
                target=self._evaluate,
                args=(output, pipeline, *args),
                kwargs=kwargs
            )
            self.subprocess = process
            process.daemon = True
            process.start()

            LOGGER.info('Joining process %s', process.pid)
            process.join()

            LOGGER.info('Terminating process %s', process.pid)
            process.terminate()
            self.subprocess = None

            result = tuple(output) if output else None

        if not result:
            raise Exception("Evaluate crashed")

        return result

    def score_pipeline(self, dataset, problem, pipeline, metrics=None, random_seed=0,
                       folds=5, stratified=False, shuffle=False):

        problem_metrics = problem['problem']['performance_metrics']
        metrics = metrics or problem_metrics
        data_params = {
            'number_of_folds': json.dumps(folds),
            'stratified': json.dumps(stratified),
            'shuffle': json.dumps(shuffle),
        }

        # Some primitives crash with a core dump that kills everything.
        # We want to isolate those.
        # This is disabled in favor of permanently using a child process
        # primitives = [
        #     step['primitive']['python_path']
        #     for step in pipeline.to_json_structure()['steps']
        # ]
        # if any(primitive in SUBPROCESS_PRIMITIVES for primitive in primitives):
        #     evaluate = self.subprocess_evaluate
        # else:
        #     evaluate = d3m_evaluate

        all_scores, all_results = self.subprocess_evaluate(
            pipeline=pipeline,
            inputs=[dataset],
            data_pipeline=self.data_pipeline,
            scoring_pipeline=self.scoring_pipeline,
            problem_description=problem,
            data_params=data_params,
            metrics=metrics,
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
            cause = failed_result.error.__cause__
            if isinstance(cause, BaseException):
                raise cause
            else:
                raise Exception(cause)

        for res in all_results:
            pipeline_run = res.pipeline_run
            if pipeline_run.run.get('phase') == 'PRODUCE':
                yaml_file = os.path.join(self.runs_dir, '{}.yml'.format(pipeline_run.get_id()))
                with open(yaml_file, 'w') as pip:
                    pipeline_run.to_yaml(file=pip)

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
        if self.subprocess:
            LOGGER.info('Terminating subprocess: %s', self.subprocess.pid)
            self.subprocess.terminate()
            self.subprocess = None

    def _timeout(self, *args, **kwargs):
        self.errors.append('STOP BY TIMEOUT')
        self.timeout_kill = True
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

    def get_data_augmentation(self, dataset, problem):
        datamart = RESTDatamart(DATAMART_URL)
        data_augmentation = problem.get('data_augmentation')
        if data_augmentation:
            LOGGER.info("DATA AUGMENTATION: Querying DataMart")
            try:
                keywords = data_augmentation[0]['keywords']
                query = DatamartQuery(keywords=keywords)
                cursor = datamart.search_with_data(query=query, supplied_data=dataset)
                LOGGER.info("DATA AUGMENTATION: Getting next page")
                page = cursor.get_next_page()
                if page:
                    result = page[0]
                    return result.serialize()

            except Exception:
                LOGGER.exception("DATA AUGMENTATION ERROR")

        # TODO: Replace this with the real DataMart query
        # if problem['id'] == 'DA_ny_taxi_demand_problem_TRAIN':
        #     LOGGER.info('DATA AUGMENTATION!!!!!!')
        #     with open(os.path.join(BASE_DIR, 'da.json')) as f:
        #         return json.dumps(json.load(f))

    def make_btb_scorer(self, dataset_name, dataset, problem, templates, metric):
        def btb_scorer(template_name, proposal):
            self.check_stop()

            pipeline = self._new_pipeline(templates[template_name], proposal)

            try:
                self.score_pipeline(dataset, problem, pipeline)
                pipeline.normalized_score = metric.normalize(pipeline.score)
                if pipeline.normalized_score > self.best_normalized:
                    self.best_normalized = pipeline.normalized_score
                    self.best_score = pipeline.score
                    self.best_pipeline = pipeline.id
                    self.best_template_name = template_name

                return pipeline.normalized_score

            finally:
                try:
                    self._save_pipeline(pipeline)
                except Exception:
                    LOGGER.exception('Error saving pipeline %s', pipeline.id)

        return btb_scorer

    def search(self, dataset, problem, timeout=None, budget=None, template_names=None):
        self.timeout = timeout
        self.timeout_kill = False
        self.budget = budget
        self.best_pipeline = None
        self.best_score = None
        self.best_normalized = -np.inf
        self.best_template_name = None
        self.found_by_name = True
        template_names = template_names or list()
        data_modality = None
        task_type = None
        task_subtype = None
        self.errors = list()

        dataset_name = problem['inputs'][0]['dataset_id']
        if dataset_name.endswith('_dataset'):
            dataset_name = dataset_name[:-len('_dataset')]

        metric = problem['problem']['performance_metrics'][0]['metric']

        data_modality, task_type, task_subtype = get_dataset_details(dataset, problem)

        # data_augmentation = self.get_data_augmentation(dataset, problem)

        LOGGER.info("Searching dataset %s: %s/%s/%s",
                    dataset_name, data_modality, task_type, task_subtype)

        try:
            self.setup_search()
            LOGGER.info("Loading the template and the tuner")
            if not template_names:
                # template_names = self._get_templates(dataset_name, data_modality, task_type)
                template_names = os.listdir(TEMPLATES_DIR)

            template_loader = LazyLoader(template_names, TEMPLATES_DIR)
            btb_scorer = self.make_btb_scorer(
                dataset_name, dataset, problem, template_loader, metric)

            session = BTBSession(template_loader, btb_scorer, max_errors=0)

            if self.budget is not None:
                while session.iterations < self.budget:
                    session.run(1)
                    last_score = list(session.proposals.values())[-1].get('score')
                    if self.ignore_errors and (last_score is None):
                        LOGGER.warning("Ignoring Errored pipeline")
                        session.iterations -= 1

            else:
                session.run()

        except KeyboardInterrupt:
            pass
        except Exception:
            LOGGER.exception("Error processing dataset %s", dataset)

        finally:
            if self.timeout and self.hard_timeout:
                signal.alarm(0)

        self.done = True

        return {
            'pipeline': self.best_pipeline,
            'cv_score': self.best_score,
            'template': self.best_template_name,
            'data_modality': data_modality,
            'task_type': task_type,
            'task_subtype': task_subtype,
            'tuning_iterations': session.iterations if session else None,
            'error': self.errors or None,
            'killed_by_timeout': self.timeout_kill,
            'pipelines_scheduled': len(template_names),
            'pipelines_tried': len([x for x in template_loader.values() if x is not None]),
            'found_by_name': self.found_by_name
        }
