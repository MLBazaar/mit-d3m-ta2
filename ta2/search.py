import json
import logging
import os
import random
import signal
import warnings
from datetime import datetime, timedelta
from multiprocessing import Manager, Process

import numpy as np
import pandas as pd
import yaml
from btb import BTBSession
from btb.tuning import StopTuning
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.runtime import DEFAULT_SCORING_PIPELINE_PATH
from d3m.runtime import evaluate as d3m_evaluate
from datamart import DatamartQuery
from datamart_rest import RESTDatamart

from ta2.loader import LazyLoader
from ta2.utils import dump_pipeline, get_dataset_details, to_dicts

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINES_DIR = os.path.join(BASE_DIR, 'pipelines')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
TEMPLATES_CSV = os.path.join(BASE_DIR, 'templates.csv')

DATAMART_URL = os.getenv('DATAMART_URL_NYU', 'https://datamart.d3m.vida-nyu.org')

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SubprocessTimeout(Exception):
    pass


class ScoringError(Exception):
    pass


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

    def _valid_template(self, template):
        try:
            path = os.path.join(TEMPLATES_DIR, template)
            with open(path, 'r') as f:
                json.load(f)
                return True

        except Exception:
            LOGGER.warning('Invalid template found: %s', path)
            return False

    def _select_templates(self, dataset_name, data_modality, task_type, templates_csv):
        templates = pd.read_csv(templates_csv)
        if 'z_score' not in templates:
            templates['z_score'] = 0
        if 'problem_type' not in templates:
            templates['problem_type'] = templates['data_modality'] + '/' + templates['task_type']

        selected = None
        if 'dataset' in templates:
            dataset_templates = templates[templates.dataset == dataset_name]
            if not dataset_templates.empty:
                dataset_templates = dataset_templates.groupby('template').z_score.max()
                selected = list(dataset_templates.sort_values(ascending=False).head(5).index)

        if not selected:
            problem_type = data_modality + '/' + task_type
            problem_templates = templates[templates.problem_type == problem_type]

            problem_templates = problem_templates.sort_values('z_score', ascending=False)
            if 'dataset' in problem_templates:
                problem_templates = problem_templates.groupby('dataset').head(3)

            z_scores = problem_templates.groupby('template').z_score.mean()
            selected = list(z_scores.sort_values(ascending=False).index)

        return list(filter(self._valid_template, selected))

    def _get_all_templates(self):
        all_templates = list(filter(self._valid_template, os.listdir(TEMPLATES_DIR)))
        return random.sample(all_templates, len(all_templates))

    def __init__(self, input_dir='input', output_dir='output', static_dir='static',
                 dump=False, hard_timeout=False, ignore_errors=False, cv_folds=5,
                 subprocess_timeout=None, max_errors=5, store_summary=False):
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
        self.folds = cv_folds
        self.subprocess_timeout = subprocess_timeout
        self.max_errors = max_errors
        self.store_summary = store_summary

    @staticmethod
    def _evaluate(out, pipeline, *args, **kwargs):
        LOGGER.info('Running d3m.runtime.evalute on pipeline %s', pipeline.id)
        results = d3m_evaluate(pipeline, *args, **kwargs)

        LOGGER.info('Returning results for %s', pipeline.id)
        out.extend(results)

    def subprocess_evaluate(self, pipeline, *args, **kwargs):
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
            process.join(self.subprocess_timeout)

            self.subprocess = None

            if process.is_alive():
                process.terminate()
                raise SubprocessTimeout('Timeout reached for subprocess {}'.format(process.pid))

            if not output:
                raise Exception("Subprocess evaluate crashed")

            return tuple(output)

    def score_pipeline(self, dataset, problem, pipeline, metrics=None, random_seed=0,
                       folds=None, stratified=False, shuffle=False, template_name=None):

        folds = folds or self.folds
        problem_metrics = problem['problem']['performance_metrics']
        metrics = metrics or problem_metrics
        data_params = {
            'number_of_folds': json.dumps(folds),
            'stratified': json.dumps(stratified),
            'shuffle': json.dumps(shuffle),
        }

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
            raise ScoringError(message)

        elif self.store_summary:
            yaml_path = os.path.join(self.runs_dir, '{}.yml'.format(pipeline.id))
            runs = [res.pipeline_run.to_json_structure() for res in all_results]
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump_all(runs, yaml_file, default_flow_style=False)

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
        self.killed = True
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

    def make_btb_scorer(self, dataset, problem, templates, metric):
        def btb_scorer(template_name, proposal):
            self.check_stop()
            self.iterations += 1
            LOGGER.info('Scoring template %s', template_name)

            pipeline = None
            status = None
            score = None
            normalized = None
            try:
                pipeline = self._new_pipeline(templates[template_name], proposal)

                self.score_pipeline(dataset, problem, pipeline)
                pipeline.normalized_score = metric.normalize(pipeline.score)
                if pipeline.normalized_score > self.best_normalized:
                    self.best_normalized = pipeline.normalized_score
                    self.best_score = pipeline.score
                    self.best_pipeline = pipeline.id
                    self.best_template_name = template_name

                LOGGER.warning('Template %s score: %s - %s', template_name,
                               pipeline.score, pipeline.normalized_score)
                status = 'SCORED'
                score = pipeline.score
                normalized = pipeline.normalized_score
                self.scored += 1
                return pipeline.normalized_score

            except SubprocessTimeout:
                self.timedout += 1
                status = 'TIMEOUT'
                raise
            except ScoringError:
                self.errored += 1
                status = 'ERROR'
                raise
            except Exception:
                self.invalid += 1
                status = 'INVALID'
                raise

            finally:
                self.summary.append({
                    'template': template_name,
                    'status': status,
                    'score': score,
                    'normalized': normalized
                })
                if pipeline:
                    pipeline_id = pipeline.id
                    try:
                        self._save_pipeline(pipeline)
                        self.summary[-1]['pipeline'] = pipeline_id
                    except Exception:
                        LOGGER.exception('Error saving pipeline %s', pipeline.id)

        return btb_scorer

    def start_session(self, template_names, dataset, problem, metric, budget):
        LOGGER.warning('Selected %s templates', len(template_names))
        template_loader = LazyLoader(template_names, TEMPLATES_DIR)
        btb_scorer = self.make_btb_scorer(dataset, problem, template_loader, metric)

        session = BTBSession(template_loader, btb_scorer, max_errors=self.max_errors)

        if budget:
            while self.spent < budget:
                session.run(1)
                last_score = list(session.proposals.values())[-1].get('score')
                if (last_score is None) and self.ignore_errors:
                    LOGGER.warning("Ignoring errored pipeline")
                else:
                    self.spent += 1

                LOGGER.warn('its: %s; sc: %s; er: %s; in: %s; ti: %s', self.iterations,
                            self.scored, self.errored, self.invalid, self.timedout)

        else:
            session.run()

    def search(self, dataset, problem, timeout=None, budget=None, templates_csv=None):
        self.timeout = timeout
        self.killed = False
        self.best_pipeline = None
        self.best_score = None
        self.best_normalized = -np.inf
        self.best_template_name = None
        self.found_by_name = True
        data_modality = None
        task_type = None
        task_subtype = None
        self.spent = 0
        self.iterations = 0
        self.scored = 0
        self.errored = 0
        self.invalid = 0
        self.timedout = 0
        self.summary = list()

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
            if not templates_csv:
                templates_csv = TEMPLATES_CSV

            template_names = self._select_templates(
                dataset_name, data_modality, task_type, templates_csv)

            if (budget is not None) and budget < 0:
                budget = len(template_names) * -budget

            try:
                self.start_session(template_names, dataset, problem, metric, budget)
            except StopTuning:
                LOGGER.warning('All selected templates failed. Falling back to the rest')
                all_templates = self._get_all_templates()
                untried_templates = [
                    template
                    for template in all_templates
                    if template not in template_names
                ]
                self.start_session(untried_templates, dataset, problem, metric, budget)

        except KeyboardInterrupt:
            pass
        except Exception:
            LOGGER.exception("Error processing dataset %s", dataset)

        finally:
            if self.timeout and self.hard_timeout:
                signal.alarm(0)

        if self.store_summary and self.summary:
            # TODO: Do this outside, in __main__.py
            # Store all the summary at once
            summary_path = os.path.join(self.output, 'summary.csv')
            self.summary = pd.DataFrame(self.summary)
            self.summary['dataset'] = dataset_name
            self.summary['data_modality'] = data_modality
            self.summary['type'] = task_type
            self.summary['subtype'] = task_subtype
            self.summary.to_csv(summary_path, index=False)

        self.done = True

        return {
            'pipeline': self.best_pipeline,
            'summary': self.summary,
            'cv_score': self.best_score,
            'template': self.best_template_name,
            'modality': data_modality,
            'type': task_type,
            'subtype': task_subtype,
            'iterations': self.iterations,
            'templates': len(template_names or []),
            'scored': self.scored,
            'errored': self.errored,
            'invalid': self.invalid,
            'timedout': self.timedout,
            'killed': self.killed,
            'found': self.found_by_name,
            'metric': metric.name.lower()
        }
