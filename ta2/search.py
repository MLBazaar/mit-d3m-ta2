import json
import logging
import os
import random
import signal
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from multiprocessing import Manager, Process

import numpy as np
import yaml
from btb.session import BTBSession
from btb.tuning.tunable import Tunable
from d3m.container.dataset import Dataset
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.problem import TaskKeyword
from d3m.runtime import DEFAULT_SCORING_PIPELINE_PATH
from d3m.runtime import evaluate as d3m_evaluate
from datamart import DatamartQuery
from datamart_rest import RESTDatamart

from ta2.template import load_template
from ta2.utils import dump_pipeline

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINES_DIR = os.path.join(BASE_DIR, 'pipelines')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
FALLBACK_PIPELINE = 'fallback_pipeline.yml'

DATAMART_URL = os.getenv('DATAMART_URL_NYU', 'https://datamart.d3m.vida-nyu.org')

TUNING_PARAMETER = 'https://metadata.datadrivendiscovery.org/types/TuningParameter'

SUBPROCESS_PRIMITIVES = [
    'd3m.primitives.natural_language_processing.lda.Fastlvm'
]

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Templates(Enum):
    # SINGLE TABLE CLASSIFICATION
    SINGLE_TABLE_CLASSIFICATION_ENC_XGB = 'single_table_classification_encoding_xgb.yml'
    # SINGLE_TABLE_CLASSIFICATION_AR_RF = 'single_table_classification_autorpi_rf.yml'
    SINGLE_TABLE_CLASSIFICATION_DFS_ROBUST_XGB = 'single_table_classification_dfs_robust_xgb.yml'
    # SINGLE_TABLE_CLASSIFICATION_DFS_XGB = 'single_table_classification_dfs_xgb.yml'
    # SINGLE_TABLE_CLASSIFICATION_GB = 'single_table_classification_gradient_boosting.yml'

    # SINGLE TABLE REGRESSION
    SINGLE_TABLE_REGRESSION_XGB = 'single_table_regression_xgb.yml'
    SINGLE_TABLE_REGRESSION_SC_XGB = 'single_table_regression_scale_xgb.yml'
    SINGLE_TABLE_REGRESSION_ENC_XGB = 'single_table_regression_encoding_xgb.yml'
    # SINGLE_TABLE_REGRESSION_DFS_XGB = 'single_table_regression_dfs_xgb.yml'
    # SINGLE_TABLE_REGRESSION_GB = 'single_table_regression_gradient_boosting.yml'

    # MISC
    SINGLE_TABLE_SEMI_CLASSIFICATION = 'single_table_semi_classification_autonbox.yml'
    # SINGLE_TABLE_CLUSTERING = 'single_table_clustering_ekss.yml'

    # MULTI TABLE
    MULTI_TABLE_CLASSIFICATION_DFS_XGB = 'multi_table_classification_dfs_xgb.yml'
    MULTI_TABLE_CLASSIFICATION_LDA_LOGREG = 'multi_table_classification_lda_logreg.yml'
    MULTI_TABLE_REGRESSION = 'multi_table_regression_dfs_xgb.yml'

    # TIMESERIES CLASSIFICATION
    TIMESERIES_CLASSIFICATION_KN = 'time_series_classification_k_neighbors_kn.yml'
    TIMESERIES_CLASSIFICATION_DSBOX_LR = 'time_series_classification_dsbox_lr.yml'
    TIMESERIES_CLASSIFICATION_LSTM_FCN = 'time_series_classification_lstm_fcn.yml'
    # TIMESERIES_CLASSIFICATION_XGB = 'time_series_classification_xgb.yml'
    # TIMESERIES_CLASSIFICATION_RF = 'time_series_classification_rf.yml'

    # IMAGE
    IMAGE_REGRESSION = 'image_regression_resnet50_xgb.yml'
    IMAGE_CLASSIFICATION = 'image_classification_resnet50_xgb.yml'
    IMAGE_OBJECT_DETECTION = 'image_object_detection_yolo.yml'

    # TEXT
    TEXT_CLASSIFICATION = 'text_classification_encoding_xgb.yml'
    TEXT_REGRESSION = 'text_regression_encoding_xgb.yml'

    # GRAPH
    # GRAPH_COMMUNITY_DETECTION_DISTIL = 'graph_community_detection_distil.yml'
    # GRAPH_LINK_PREDICTION = 'graph_link_prediction_distil.yml'
    # GRAPH_MATCHING = 'graph_matching.yml'
    # GRAPH_MATCHING_JHU = 'graph_matching_jhu.yml'


def sanitize_hyperparameters(hyperparams):
    sanitized = {
        (step, name): value
        for step, tunable_hp in hyperparams.items()
        for name, value in tunable_hp.items()
    }

    for tunable_hp in sanitized.values():
        if tunable_hp['type'] == 'string':
            tunable_hp['type'] = 'str'
        if tunable_hp['type'] == 'integer':
            tunable_hp['type'] = 'int'
        if tunable_hp['type'] == 'boolean':
            tunable_hp['type'] = 'bool'

    return sanitized


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


FILE_COLLECTION = 'https://metadata.datadrivendiscovery.org/types/FilesCollection'
GRAPH = 'https://metadata.datadrivendiscovery.org/types/Graph'
EDGE_LIST = 'https://metadata.datadrivendiscovery.org/types/EdgeList'


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

    def _get_templates(self, data_modality, task_type):
        LOGGER.info("Loading template for data modality %s and task type %s",
                    data_modality, task_type)

        templates = [Templates.SINGLE_TABLE_CLASSIFICATION_ENC_XGB]

        if data_modality == 'single_table':
            if task_type == TaskKeyword.CLASSIFICATION.name.lower():
                templates = [
                    Templates.SINGLE_TABLE_CLASSIFICATION_ENC_XGB,
                    Templates.SINGLE_TABLE_CLASSIFICATION_DFS_ROBUST_XGB,
                ]
            elif task_type == TaskKeyword.REGRESSION.name.lower():
                templates = [
                    Templates.SINGLE_TABLE_REGRESSION_XGB,
                    Templates.SINGLE_TABLE_REGRESSION_SC_XGB,
                    Templates.SINGLE_TABLE_REGRESSION_ENC_XGB,
                ]
            elif task_type == TaskKeyword.COLLABORATIVE_FILTERING.name.lower():
                templates = [
                    Templates.SINGLE_TABLE_REGRESSION_XGB,
                    Templates.SINGLE_TABLE_REGRESSION_SC_XGB,
                    Templates.SINGLE_TABLE_REGRESSION_ENC_XGB,
                ]
            elif task_type == TaskKeyword.TIME_SERIES_FORECASTING.name.lower():
                templates = [
                    Templates.SINGLE_TABLE_REGRESSION_XGB,
                    Templates.SINGLE_TABLE_REGRESSION_SC_XGB,
                    Templates.SINGLE_TABLE_REGRESSION_ENC_XGB,
                ]
            elif task_type == TaskKeyword.SEMISUPERVISED_CLASSIFICATION.name.lower():
                templates = [Templates.SINGLE_TABLE_SEMI_CLASSIFICATION]

        if data_modality == 'multi_table':
            if task_type == TaskKeyword.CLASSIFICATION.name.lower():
                templates = [
                    Templates.MULTI_TABLE_CLASSIFICATION_LDA_LOGREG,
                    Templates.MULTI_TABLE_CLASSIFICATION_DFS_XGB,
                ]
            elif task_type == TaskKeyword.REGRESSION.name.lower():
                templates = [Templates.MULTI_TABLE_REGRESSION]
        elif data_modality == 'text':
            if task_type == TaskKeyword.CLASSIFICATION.name.lower():
                templates = [Templates.TEXT_CLASSIFICATION]
            elif task_type == TaskKeyword.REGRESSION.name.lower():
                templates = [Templates.TEXT_REGRESSION]

        if data_modality == 'timeseries':
            templates = [
                Templates.TIMESERIES_CLASSIFICATION_KN,
                Templates.TIMESERIES_CLASSIFICATION_DSBOX_LR,
                Templates.TIMESERIES_CLASSIFICATION_LSTM_FCN
            ]
            # if task_type == TaskKeyword.CLASSIFICATION.name.lower():
            #     template = Templates.TIMESERIES_CLASSIFICATION
            # elif task_type == TaskKeyword.REGRESSION.name.lower():
            #     template = Templates.TIMESERIES_REGRESSION
        elif data_modality == 'image':
            if task_type == TaskKeyword.CLASSIFICATION.name.lower():
                templates = [Templates.IMAGE_CLASSIFICATION]
            elif task_type == TaskKeyword.REGRESSION.name.lower():
                templates = [Templates.IMAGE_REGRESSION]
            elif task_type == TaskKeyword.OBJECT_DETECTION.name.lower():
                templates = [Templates.IMAGE_OBJECT_DETECTION]

        if data_modality == 'graph':
            if task_type == TaskKeyword.COMMUNITY_DETECTION.name.lower():
                templates = [Templates.GRAPH_COMMUNITY_DETECTION]
            elif task_type == TaskKeyword.VERTEX_CLASSIFICATION.name.lower():
                templates = [Templates.SINGLE_TABLE_CLASSIFICATION_ENC_XGB]

        return [template.value for template in templates]

    def __init__(self, input_dir='input', output_dir='output', static_dir='static',
                 dump=False, hard_timeout=False):
        self.input = input_dir
        self.output = output_dir
        self.static = static_dir
        self.dump = dump
        self.hard_timeout = hard_timeout
        self.subprocess = None

        self.ranked_dir = os.path.join(self.output, 'pipelines_ranked')
        self.scored_dir = os.path.join(self.output, 'pipelines_scored')
        self.searched_dir = os.path.join(self.output, 'pipelines_searched')
        os.makedirs(self.ranked_dir, exist_ok=True)
        os.makedirs(self.scored_dir, exist_ok=True)
        os.makedirs(self.searched_dir, exist_ok=True)

        self.solutions = list()
        self.data_pipeline = self._load_pipeline('kfold_pipeline.yml')
        self.scoring_pipeline = self._load_pipeline(DEFAULT_SCORING_PIPELINE_PATH)

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
        primitives = [
            step['primitive']['python_path']
            for step in pipeline.to_json_structure()['steps']
        ]
        if any(primitive in SUBPROCESS_PRIMITIVES for primitive in primitives):
            evaluate = self.subprocess_evaluate
        else:
            evaluate = d3m_evaluate

        all_scores, all_results = evaluate(
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
        # if self.subprocess:
        #     LOGGER.info('Terminating subprocess: %s', self.subprocess.pid)
        #     self.subprocess.terminate()
        #     self.subprocess = None

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
            self.iterations += 1

            pipeline = self._new_pipeline(templates[template_name], proposal)
            params = '\n'.join('{}: {}'.format(k, v) for k, v in proposal.items())

            LOGGER.warn('Scoring pipeline %s - %s: %s\n%s',
                        self.iterations, template_name, pipeline.id, params)

            try:
                self.score_pipeline(dataset, problem, pipeline)
                pipeline.normalized_score = metric.normalize(pipeline.score)

            except Exception as ex:
                LOGGER.exception('Error scoring pipeline %s for dataset %s',
                                 pipeline.id, dataset_name)

                error = '{}: {}'.format(type(ex).__name__, ex)
                self.errors.append(error)
                max_errors = min(len(templates), self.budget or np.inf)
                if len(self.errors) >= max_errors:
                    raise Exception(self.errors)

                pipeline.score = None
                pipeline.normalized_score = 0.0

            LOGGER.info('Pipeline %s score: %s - %s',
                        pipeline.id, pipeline.score, pipeline.normalized_score)

            try:
                self._save_pipeline(pipeline)

            except Exception:
                LOGGER.exception('Error saving pipeline %s', pipeline.id)

            if pipeline.normalized_score > self.best_normalized:
                LOGGER.warn('New best pipeline found: %s! %s is better than %s',
                            template_name, pipeline.score, self.best_score)

                self.best_pipeline = pipeline.id
                self.best_score = pipeline.score
                self.best_normmalized = pipeline.normalized_score
                self.best_template_name = template_name

            return pipeline.normalized_score

        return btb_scorer

    @staticmethod
    def _get_tunables_templates(template_names):
        tunables = {}
        templates = {}

        for template_name in template_names:
            template, tunable_hp = load_template(template_name)
            templates[template_name] = template
            tunables[template_name] = Tunable.from_dict(sanitize_hyperparameters(tunable_hp))

        return tunables, templates

    def _get_fallback_pipeline(self, data_modality, task_type):

        fallback_name = '{}_{}_fallback'.format(data_modality, task_type)
        fallback_path = os.path.dirname(os.path.abspath(__file__))
        fallback_path = os.path.join(fallback_path, 'pipelines/fallback_pipelines')
        fallback_pipeline = None

        for pipeline in os.listdir(fallback_path):
            if pipeline.startswith(fallback_name):
                fallback_pipeline = os.path.join(fallback_path, pipeline)
                break

        if fallback_pipeline is None:
            LOGGER.info('No fallback pipeline found for %s %s' % data_modality, task_type)
            return None

        with open(fallback_pipeline) as pipeline:
            if fallback_pipeline.endswith('yml'):
                data = yaml.load(pipeline)

            else:
                data = json.load(pipeline)

        return Pipeline.from_json_structure(data)

    def search(self, dataset_path, problem, timeout=None, budget=None, template_names=None):
        self.timeout = timeout
        self.budget = budget
        self.best_pipeline = None
        self.best_score = None
        self.best_normalized = 0
        self.best_template_name = None
        self.iterations = 0
        template_names = template_names or list()
        data_modality = None
        task_type = None
        task_subtype = None
        self.errors = list()

        dataset_name = problem['inputs'][0]['dataset_id']
        if dataset_name.endswith('_dataset'):
            dataset_name = dataset_name[:-len('_dataset')]

        dataset = Dataset.load(dataset_path)
        metric = problem['problem']['performance_metrics'][0]['metric']

        data_modality = detect_data_modality(dataset_path[7:])
        task_type = problem['problem']['task_keywords'][0].name.lower()
        task_subtype = problem['problem']['task_keywords'][1].name.lower()

        self.fallback = self._get_fallback_pipeline(data_modality, task_type)

        # data_augmentation = self.get_data_augmentation(dataset, problem)

        LOGGER.info("Searching dataset %s: %s/%s/%s",
                    dataset_name, data_modality, task_type, task_subtype)

        try:
            self.setup_search()

            if self.fallback:

                self.score_pipeline(dataset, problem, self.fallback)
                self.fallback.normalized_score = metric.normalize(self.fallback.score)
                self._save_pipeline(self.fallback)
                self.best_pipeline = self.fallback.id
                self.best_score = self.fallback.score
                self.best_template_name = FALLBACK_PIPELINE
                self.best_normalized = self.fallback.normalized_score

                LOGGER.warn("Fallback pipeline score: %s - %s",
                            self.fallback.score, self.fallback.normalized_score)

            LOGGER.info("Loading the template and the tuner")
            if not template_names:
                template_names = self._get_templates(data_modality, task_type)

            tunables, templates = self._get_tunables_templates(template_names)
            btb_scorer = self.make_btb_scorer(dataset_name, dataset, problem, templates, metric)

            session = BTBSession(tunables, btb_scorer)

            if self.budget is not None:
                session.run(self.budget)
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
            'tuning_iterations': self.iterations,
            'error': self.errors or None
        }
