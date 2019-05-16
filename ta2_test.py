import argparse
import logging
import os
import time

import pandas as pd
import tabulate

from d3m.container.dataset import Dataset
from d3m.metadata.base import Context
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import Problem
from d3m.runtime import Runtime, score

from ta2 import logging_setup
from ta2.search import PipelineSearcher


def load_dataset(root_path, phase, inner_phase=None):
    inner_phase = inner_phase or phase
    path = os.path.join(root_path, phase, 'dataset_' + inner_phase, 'datasetDoc.json')
    return Dataset.load(dataset_uri='file://' + os.path.abspath(path))


def load_problem(root_path, phase):
    path = os.path.join(root_path, phase, 'problem_' + phase, 'problemDoc.json')
    return Problem.load(problem_uri='file://' + os.path.abspath(path))


def load_pipeline(pipeline_path):
    with open(pipeline_path, 'r') as pipeline_file:
        if pipeline_path.endswith('.json'):
            return Pipeline.from_json(pipeline_file)
        else:
            return Pipeline.from_yaml(pipeline_file)


def search(dataset_root, problem, args):

    pps = PipelineSearcher(args.input, args.output, dump=True)

    return pps.search(problem, timeout=args.timeout, budget=args.budget)


def score_pipeline(dataset_root, problem, pipeline_path):
    train_dataset = load_dataset(dataset_root, 'TRAIN')
    test_dataset = load_dataset(dataset_root, 'SCORE', 'TEST')
    pipeline = load_pipeline(pipeline_path)

    # Creating an instance on runtime with pipeline description and problem description.
    runtime = Runtime(
        pipeline=pipeline,
        problem_description=problem,
        context=Context.TESTING
    )

    print("Fitting the pipeline")
    fit_results = runtime.fit(inputs=[train_dataset])
    fit_results.check_success()

    # Producing results using the fitted pipeline.
    print("Producing predictions")
    produce_results = runtime.produce(inputs=[test_dataset])
    produce_results.check_success()

    predictions = produce_results.values['outputs.0']
    metrics = problem['problem']['performance_metrics']

    print("Computing the score")
    scoring_pipeline = load_pipeline('ta2/pipelines/scoring_pipeline.yml')
    scores, scoring_pipeline_run = score(
        scoring_pipeline, problem, predictions, [test_dataset], metrics,
        context=Context.TESTING, random_seed=0,
    )
    return scores.iloc[0].value


def box_print(message):
    print('#' * len(message))
    print(message)
    print('#' * len(message))


def process_dataset(dataset, args):
    start_time = time.time()

    box_print("Processing dataset {}".format(dataset))
    dataset_root = os.path.join(args.input, dataset)
    problem = load_problem(dataset_root, 'TRAIN')

    print("Searching Pipeline for dataset {}".format(dataset))
    result = search(dataset_root, problem, args)
    best_id = result['pipeline']
    best_score = result['score']
    template = result['template']
    data_modality = result['data_modality']
    task_type = result['task_type']

    if best_id is None or best_score is None:
        raise ValueError('Unsupported problem')

    best_path = os.path.join(args.output, 'pipelines_ranked', best_id + '.json')
    box_print("Best Pipeline: {} - CV Score: {}".format(best_id, best_score))

    test_score = score_pipeline(dataset_root, problem, best_path)
    box_print("Test Score for pipeline {}: {}".format(best_id, test_score))

    end_time = time.time()

    return {
        'dataset': dataset,
        'template': template,
        'cv_score': best_score,
        'test_score': test_score,
        'elapsed_time': end_time - start_time,  # seconds
        'tuning_iterations': args.budget,
        'data_modality': data_modality,
        'task_type': task_type
    }


def process_datasets(args):
    results = list()
    for d in args.dataset:
        try:
            results.append(process_dataset(d, args))
        except ValueError:
            continue

    return pd.DataFrame(
        results,
        columns=['dataset', 'template', 'cv_score', 'test_score',
                 'elapsed_time', 'tuning_iterations', 'data_modality', 'task_type']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TA2')
    parser.add_argument('-i', '--input', default='input',
                        help='Path to the datsets root folder')
    parser.add_argument('-o', '--output', default='output',
                        help='Path to the folder where outputs will be stored')
    parser.add_argument('-b', '--budget', type=int,
                        help='Maximum number of tuning iterations to perform')
    parser.add_argument('-t', '--timeout', type=int,
                        help='Maximum time allowed for the tuning, in number of seconds')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity')
    parser.add_argument('-l', '--logfile', type=str, nargs='?',
                        help='Path to the logging file. If not given, log to stdout')
    parser.add_argument('dataset', nargs='+', help='Name of the dataset to use for the test')
    parser.add_argument('-r', '--report', type=str, nargs='?',
                        help='Path to the CSV file where scores will be dumped. If not given, print to stdout')
    args = parser.parse_args()

    logging_setup(args.verbose, args.logfile)
    logging.getLogger("d3m.metadata.pipeline_run").setLevel(logging.ERROR)

    report = process_datasets(args)

    if args.report:
        # dump to file
        report.to_csv(args.report, index=False)
    else:
        # print to stdout
        print(tabulate.tabulate(
            report,
            showindex=False,
            tablefmt='github',
            headers=report.columns
        ))
