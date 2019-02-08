import argparse
import logging
import os

from d3m.metadata.problem import parse_problem_description

from ta2 import logging_setup
from ta2.search import PipelineSearcher


def load_problem(root_path, phase):
    path = os.path.join(root_path, phase, 'problem_' + phase, 'problemDoc.json')
    return parse_problem_description(path)


def search(dataset, args):
    dataset_root = os.path.join(args.input, dataset)

    problem = load_problem(dataset_root, 'TRAIN')

    pps = PipelineSearcher(args.input, args.output, dump=True)

    pps.search(problem, timeout=args.timeout, budget=args.budget)


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

    args = parser.parse_args()

    logging_setup(args.verbose, args.logfile)
    logging.getLogger("d3m.metadata.pipeline_run").setLevel(logging.ERROR)

    for dataset in args.dataset:
        search(dataset, args)
