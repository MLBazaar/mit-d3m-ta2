import argparse
import gc
import logging
import os
import socket
import subprocess
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import tabulate

import ta2
from ta2.standalone import process_dataset
from ta2.ta3.client import TA3Client
from ta2.ta3.server import serve
from ta2.utils import get_datasets, logging_setup

LOGGER = logging.getLogger(__name__)
DOCKER_IMAGE = 'mlbazaar/mit-d3m-ta2:{}'.format(ta2.__version__)
HOSTNAME = socket.gethostname()


RESULTS_COLUMNS = [
    'dataset',
    'template',
    'cv_score',
    'test_score',
    'metric',
    'elapsed',
    'templates',
    'iterations',
    'scored',
    'errored',
    'invalid',
    'timedout',
    'killed',
    'data_modality',
    'task_type',
    'host',
    'timestamp'
]
SUMMARY_COLUMNS = [
    'template',
    'pipeline',
    'status',
    'score',
    'normalized',
    'dataset',
    'data_modality',
    'task_type'
]


DOCKER_BUILD = [
    'docker',
    'build',
    '--build-arg',
    'UID={}'.format(os.getuid()),
    '-t',
    DOCKER_IMAGE,
    '.'
]
DOCKER_CMD_BASE = [
    'docker',
    'run',
    '-ti',
    '--rm',
    '-u{}:{}'.format(os.getuid(), os.getgid()),
    '--hostname',
    HOSTNAME,
    '-v{}:/user_dev/'.format(os.getcwd())
]
DOCKER_ARGS = [
    '-I/input',
    '-O/output',
    '-S/static',
]
JUPYTER_CMDS = [
    'jupyter',
    'notebook',
    '--ip',
    '0.0.0.0',
    '--NotebookApp.token=""',
    '--allow-root'
]


def _get_docker_base(args, jupyter=False):
    if args.build:
        subprocess.run(DOCKER_BUILD)

    cmd = DOCKER_CMD_BASE.copy()

    cmd.append('-v{}:/input'.format(args.input))
    cmd.append('-v{}:/output'.format(args.output))
    cmd.append('-v{}:/static'.format(args.static))

    if getattr(args, 'port', None):
        cmd.append('-p{0}:{0}'.format(args.port))

    cmd.append(DOCKER_IMAGE)
    return cmd


def _standalone_docker(args):
    """Runs docker through subprocess."""
    docker_cmd = _get_docker_base(args)
    docker_cmd.append('ta2')
    docker_cmd.extend(sys.argv[1:])
    docker_cmd.append('-enative')
    docker_cmd += DOCKER_ARGS

    LOGGER.info(' '.join(docker_cmd))
    subprocess.run(docker_cmd)


def _start_report(report_path, columns):
    report = pd.DataFrame(columns=columns)
    report.to_csv(report_path, index=False)

    return report


def _append_report(data, columns, path, **extra_columns):
    report = pd.DataFrame(data, columns=columns)

    for key, value in extra_columns.items():
        report[key] = value

    report.to_csv(path, mode='a', header=False, index=False)

    return report


def _standalone_native(args):
    """Run in native mode."""

    args.results = args.results or os.path.join(args.output, 'results.csv')
    args.summary = args.summary or os.path.join(args.output, 'summary.csv')

    results = _start_report(args.results, RESULTS_COLUMNS)
    _start_report(args.summary, SUMMARY_COLUMNS)

    datasets = get_datasets(args.input, args.dataset, args.data_modality, args.task_type)
    for dataset, problem, data_modality, task_type in datasets:
        extra_columns = {
            'data_modality': data_modality,
            'task_type': task_type
        }
        try:
            output_path = os.path.join(args.output, dataset.name)
            os.makedirs(output_path, exist_ok=True)

            result = process_dataset(
                dataset,
                problem,
                args.input,
                output_path,
                args.static,
                args.ignore_errors,
                args.folds,
                args.subprocess_timeout,
                args.max_errors,
                args.timeout,
                args.budget,
                args.templates_csv,
            )
            summary = result.pop('summary', None)
            if summary:
                _append_report(summary, SUMMARY_COLUMNS, args.summary, **extra_columns)

            gc.collect()

        except Exception as ex:
            LOGGER.exception("Error processing dataset %s", dataset.name)
            traceback.print_exc()
            result = {
                'dataset': dataset.name,
                'error': '{}: {}'.format(type(ex).__name__, ex)
            }

        extra_columns.update({
            'dataset': dataset.name,
            'template': result.get('template', '')[0:12],
            'host': HOSTNAME,
            'timestamp': datetime.utcnow(),
        })

        results = results.append(
            _append_report([result], RESULTS_COLUMNS, args.results, **extra_columns),
            sort=False,
            ignore_index=True
        )

    if results.empty:
        print("No matching datasets found")
        sys.exit(1)

    print(tabulate.tabulate(
        results[RESULTS_COLUMNS],
        showindex=False,
        tablefmt='github',
        headers=RESULTS_COLUMNS
    ))


def _ta2_standalone(args):
    if args.environment == 'docker':
        _standalone_docker(args)
    else:
        _standalone_native(args)


def _ta3_test_dataset(client, dataset, timeout):
    print('### Testing dataset {} ###'.format(dataset))

    print('### {} => client.SearchSolutions("{}")'.format(dataset, dataset))
    search_response = client.search_solutions(dataset, timeout)
    search_id = search_response.search_id

    print('### {} => client.GetSearchSolutionsResults("{}")'.format(dataset, search_id))
    solutions = client.get_search_solutions_results(search_id, 2)
    solution_id = solutions[-1].solution_id

    print('### {} => client.StopSearchSolutions("{}")'.format(dataset, solution_id))
    client.stop_search_solutions(search_id)

    print('### {} => client.DescribeSolution("{}")'.format(dataset, search_id))
    client.describe_solution(solution_id)

    print('### {} => client.ScoreSolution("{}")'.format(dataset, solution_id))
    score_response = client.score_solution(solution_id, dataset)
    request_id = score_response.request_id

    print('### {} => client.GetScoreSolutionsResults("{}")'.format(dataset, request_id))
    score_solution_results = client.get_score_solution_results(request_id)

    print('### {} => client.FitSolution("{}")'.format(dataset, solution_id))
    fit_response = client.fit_solution(solution_id, dataset)
    request_id = fit_response.request_id

    print('### {} => client.GetFitSolutionsResults("{}")'.format(dataset, request_id))
    fitted_solutions = client.get_fit_solution_results(request_id)
    fitted_solution_id = fitted_solutions[-1].fitted_solution_id

    print('### {} => client.ProduceSolution("{}")'.format(dataset, fitted_solution_id))
    produce_response = client.produce_solution(fitted_solution_id, dataset)
    request_id = produce_response.request_id

    print('### {} => client.GetProduceSolutionsResults("{}")'.format(dataset, request_id))
    client.get_produce_solution_results(request_id)

    print('### {} => client.SolutionExport("{}")'.format(dataset, fitted_solution_id))
    client.solution_export(fitted_solution_id, 1)

    print('### {} => client.EndSearchSolutions("{}")'.format(dataset, search_id))
    client.end_search_solutions(search_id)

    return np.mean([
        score.value.raw.double
        for score in score_solution_results.scores
    ])


def _ta3_test(args):
    local_input = args.input
    remote_input = '/input' if args.docker else args.input
    client = TA3Client(args.port, local_input, remote_input)

    print('### Hello ###')
    client.hello()

    if args.all:
        args.dataset = os.listdir(args.input)

    results = list()
    for dataset in args.dataset:
        try:
            score = _ta3_test_dataset(client, dataset, args.timeout / 60)
            results.append({
                'dataset': dataset,
                'score': score
            })
        except Exception as ex:
            results.append({
                'dataset': dataset,
                'score': 'ERROR'
            })
            print('TA3 Error on dataset {}: {}'.format(dataset, ex))

    results = pd.DataFrame(results)
    print(tabulate.tabulate(
        results[['dataset', 'score']],
        showindex=False,
        tablefmt='github',
        headers=['dataset', 'score']
    ))


def _server(args):
    input_dir = args.input or os.getenv('D3MINPUTDIR', 'input')
    output_dir = args.output or os.getenv('D3MOUTPUTDIR', 'output')
    timeout = args.timeout or os.getenv('D3MTIMEOUT', 600)

    try:
        timeout = int(timeout)
    except ValueError:
        # FIXME This is just to be sure that it does not crash
        timeout = 600

    serve(args.port, input_dir, output_dir, args.static, timeout, args.debug)


def _jupyter_docker(args):
    docker_cmd = _get_docker_base(args)
    docker_cmd.extend(JUPYTER_CMDS)

    LOGGER.info(' '.join(docker_cmd))
    subprocess.run(docker_cmd)


def _jupyter_native(args):
    subprocess.run('jupyter-notebook')


def _jupyter(args):
    if args.environment == 'docker':
        _jupyter_docker(args)
    else:
        _jupyter_native(args)


def parse_args():

    # Logging
    logging_args = argparse.ArgumentParser(add_help=False)
    logging_args.add_argument('-v', '--verbose', action='count', default=0,
                              help='Be verbose. Use -vv for increased verbosity')
    logging_args.add_argument('-l', '--logfile', type=str, nargs='?',
                              help='Path to the logging file.')
    logging_args.add_argument('-q', '--quiet', action='store_false', dest='stdout',
                              help='Be quiet. Do not log to stdout.')

    # IO Specification
    io_args = argparse.ArgumentParser(add_help=False)
    io_args.add_argument('-I', '--input', default='input',
                         help='Path to the datsets root folder')
    io_args.add_argument('-O', '--output', default='output',
                         help='Path to the folder where outputs will be stored')
    io_args.add_argument('-S', '--static', default='static', type=str,
                         help='Path to a directory with static files required by primitives')

    # Datasets
    dataset_args = argparse.ArgumentParser(add_help=False)
    dataset_args.add_argument('-A', '--all', action='store_true',
                              help='Process all the datasets found in the input folder')
    dataset_args.add_argument('-M', '--data-modality',
                              help='Filter datasets by data modality.')
    dataset_args.add_argument('-T', '--task-type',
                              help='Filter datasets by task type.')
    dataset_args.add_argument('dataset', nargs='*',
                              help='Name of the dataset to use for the test')

    # Search Configuration
    search_args = argparse.ArgumentParser(add_help=False)
    search_args.add_argument('-t', '--timeout', type=int,
                             help='Maximum time allowed for the tuning, in number of seconds')

    # TA3-TA2 Common Args
    ta3_args = argparse.ArgumentParser(add_help=False)
    ta3_args.add_argument('-p', '--port', type=int, default=45042,
                          help='Port to use, both for client and server.')

    # Environment
    environment_args = argparse.ArgumentParser(add_help=False)
    environment_args.add_argument('-e', '--environment', default='docker',
                                  choices=['docker', 'native'],
                                  help='Execution environment mode.')
    environment_args.add_argument('-B', '--build', action='store_true',
                                  help='Build the docker image.')

    parser = argparse.ArgumentParser(
        description='TA2 Command Line Interface',
    )

    subparsers = parser.add_subparsers(title='mode', dest='mode', help='Mode of operation.')
    subparsers.required = True
    parser.set_defaults(mode=None)

    # standalone Mode
    standalone_parents = [logging_args, io_args, search_args, dataset_args, environment_args]
    standalone_parser = subparsers.add_parser('standalone', parents=standalone_parents,
                                              help='Run TA2 in Standalone Mode.')
    standalone_parser.set_defaults(mode=_ta2_standalone)
    standalone_parser.add_argument(
        '-r', '--results',
        help='Path to the CSV file where the results will be dumped.')
    standalone_parser.add_argument(
        '-s', '--summary',
        help='Path to the CSV file where search summary will be dumped.')
    standalone_parser.add_argument(
        '-b', '--budget', type=int,
        help='Maximum number of tuning iterations to perform')
    standalone_parser.add_argument(
        '-i', '--ignore-errors', action='store_true',
        help='Ignore errors when counting tuning iterations.')
    standalone_parser.add_argument(
        '-c', '--templates-csv', help='Path to the templates csv file to use.')
    standalone_parser.add_argument(
        '-f', '--folds', type=int, default=5,
        help='Number of folds to use for cross validation')
    standalone_parser.add_argument(
        '-p', '--subprocess-timeout', type=int,
        help='Maximum time allowed per pipeline execution, in seconds')
    standalone_parser.add_argument(
        '-m', '--max-errors', type=int, default=5,
        help='Maximum amount of errors per template.')

    # TA3 Mode
    ta3_parents = [logging_args, io_args, search_args, ta3_args, dataset_args, environment_args]
    ta3_parser = subparsers.add_parser('ta3', parents=ta3_parents,
                                       help='Run TA3-TA2 API Test.')
    ta3_parser.set_defaults(mode=_ta3_test)
    ta3_parser.add_argument('--server', action='store_true', help=(
        'Start a server instance in background.'
    ))
    ta3_parser.add_argument('--docker', action='store_true', help=(
        'Adapt input paths to work with a dockerized TA2.'
    ))

    # Server Mode
    server_parents = [logging_args, io_args, ta3_args, search_args, environment_args]
    server_parser = subparsers.add_parser('server', parents=server_parents,
                                          help='Start a TA3-TA2 server.')
    server_parser.set_defaults(mode=_server)
    server_parser.add_argument(
        '--debug', action='store_true',
        help='Start the server in sync mode. Needed for debugging.'
    )

    # Jupyter Mode
    jupyter_parents = [logging_args, io_args, environment_args]
    jupyter_parser = subparsers.add_parser('jupyter', parents=jupyter_parents,
                                           help='Start jupyter-notebook.')
    jupyter_parser.set_defaults(mode=_jupyter)
    jupyter_parser.add_argument('-p', '--port', type=int, default=8888,
                                help='Port to use for the jupyter notebook.')

    # ########## #
    # Parse Args #
    # ########## #
    args = parser.parse_args()

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    args.static = os.path.abspath(args.static)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if args.logfile:
        if not os.path.isabs(args.logfile):
            args.logfile = os.path.join(args.output, args.logfile)

        logdir = os.path.dirname(args.logfile)
        os.makedirs(logdir, exist_ok=True)

    if not getattr(args, 'all', True) and not args.dataset:
        print('ERROR: provide at least one dataset name or set --all')
        sys.exit(1)

    logging_setup(args.verbose, args.logfile, stdout=args.stdout)
    logging.getLogger("d3m").setLevel(logging.ERROR)
    logging.getLogger("redirect").setLevel(logging.CRITICAL)

    return args


def main():
    args = parse_args()
    args.mode(args)


if __name__ == '__main__':
    main()
