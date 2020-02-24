import logging
import os
from datetime import datetime

import pandas as pd
from d3m.metadata.base import Context
from d3m.runtime import DEFAULT_SCORING_PIPELINE_PATH, Runtime, score

from ta2.core import TA2Core
from ta2.utils import box_log, load_dataset, load_pipeline

LOGGER = logging.getLogger(__name__)


def _to_yaml_run(pipeline_run, output_path):
    run_dir = os.path.join(output_path, 'pipeline_runs')
    run_dir = os.path.join(run_dir, '{}.yml'.format(pipeline_run.get_id()))
    with open(run_dir, 'w') as output_file:
        pipeline_run.to_yaml(file=output_file)


def _score_pipeline(dataset, problem, pipeline_path, static=None, output_path=None):
    pipeline = load_pipeline(pipeline_path)

    # Creating an instance on runtime with pipeline description and problem description.
    runtime = Runtime(
        pipeline=pipeline,
        problem_description=problem,
        context=Context.EVALUATION,
        volumes_dir=static,
    )

    LOGGER.info("Fitting pipeline %s", pipeline_path)
    fit_results = runtime.fit(inputs=[dataset])
    fit_results.check_success()

    dataset_doc_path = dataset.metadata.query(())['location_uris'][0]
    dataset_root = dataset_doc_path[:-len('/TRAIN/dataset_TRAIN/datasetDoc.json')]
    test_dataset = load_dataset(dataset_root, 'SCORE', 'TEST')

    # Producing results using the fitted pipeline.
    LOGGER.info("Producing predictions for pipeline %s", pipeline_path)
    produce_results = runtime.produce(inputs=[test_dataset])
    produce_results.check_success()

    predictions = produce_results.values['outputs.0']
    metrics = problem['problem']['performance_metrics']

    LOGGER.info("Computing the score for pipeline %s", pipeline_path)
    scoring_pipeline = load_pipeline(DEFAULT_SCORING_PIPELINE_PATH)
    scores, scoring_pipeline_run = score(
        scoring_pipeline=scoring_pipeline,
        problem_description=problem,
        predictions=predictions,
        score_inputs=[test_dataset],
        metrics=metrics,
        context=Context.EVALUATION,
        random_seed=0,
    )

    evaluated_pipeline_run = produce_results.pipeline_run
    evaluated_pipeline_run.is_standard_pipeline = True
    evaluated_pipeline_run.set_scores(scores, metrics)
    evaluated_pipeline_run.set_scoring_pipeline_run(scoring_pipeline_run.pipeline_run, [dataset])

    _to_yaml_run(evaluated_pipeline_run, output_path)

    return scores.iloc[0].value


def _select_candidates(summary):
    summary = pd.DataFrame(summary)
    summary = summary[summary.status == 'SCORED']
    summary = summary[['template', 'pipeline', 'score', 'normalized']]
    candidates = summary.sort_values('normalized', ascending=False).head(20)
    candidates['pipeline'] += '.json'
    return candidates


def process_dataset(dataset, problem, input_path, output_path, static_path,
                    hard_timeout, ignore_errors, folds, subprocess_timeout, max_errors,
                    timeout, budget, templates_csv):
    box_log("Processing dataset {}".format(dataset.name), True)
    try:
        start_ts = datetime.utcnow()
        ta2_core = TA2Core(
            input_path,
            output_path,
            static_path,
            dump=True,
            hard_timeout=hard_timeout,
            ignore_errors=ignore_errors,
            cv_folds=folds,
            subprocess_timeout=subprocess_timeout,
            max_errors=max_errors,
            store_summary=True
        )
        result = ta2_core.search(dataset, problem, timeout, budget, templates_csv)

        result['elapsed'] = datetime.utcnow() - start_ts

    except Exception as ex:
        result = {
            'error': '{}: {}'.format(type(ex).__name__, ex),
        }
    else:
        try:
            candidates = _select_candidates(result['summary'])
            if candidates.empty:
                box_log('No valid pipelines found for dataset {}'.format(dataset.name))
            else:
                ranked_path = os.path.join(output_path, 'pipelines_ranked')
                test_scores = list()
                for _, candidate in candidates.iterrows():
                    try:
                        pipeline = candidate.pipeline
                        pipeline_path = os.path.join(ranked_path, pipeline)
                        test_score = _score_pipeline(dataset, problem, pipeline_path,
                                                     static_path, output_path)
                        test_scores.append(test_score)
                    except Exception:
                        test_scores.append(None)

                candidates['test_score'] = test_scores
                candidates = candidates.sort_values('test_score', ascending=False)

                best = candidates.iloc[0]
                result['test_score'] = best.test_score
                result['template'] = best.template
                result['cv_score'] = best.score
                box_log('Best pipelines for dataset {}:\n{}'.format(
                    dataset.name, candidates.to_string()))

        except Exception as ex:
            LOGGER.exception('Error while testing the winner pipeline')
            result['error'] = 'TEST Error: {}'.format(ex)

    return result
