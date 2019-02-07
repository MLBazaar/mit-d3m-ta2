import argparse
import logging
import os

from d3m.metadata.problem import parse_problem_description

from ta2 import logging_setup
from ta2.search import PipelineSearcher


def load_problem(root_path, phase):
    path = os.path.join(root_path, phase, 'problem_' + phase, 'problemDoc.json')
    return parse_problem_description(path)


def search(args):
    dataset_root = args.datasets_root + args.dataset

    problem = load_problem(dataset_root, 'TRAIN')

    pps = PipelineSearcher(args.input, args.output, args.pipelines)

    pps.search(problem, timeout=args.timeout, budget=args.budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TA2')
    parser.add_argument('-i', '--input', default='input')
    parser.add_argument('-o', '--output', default='output')
    parser.add_argument('-p', '--pipelines')
    parser.add_argument('-b', '--budget', type=int)
    parser.add_argument('-t', '--timeout', type=int)
    parser.add_argument('-d', '--datasets-root', default='/opt/datasets/all/3.2.0/')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-l', '--logfile', type=str, nargs='?')
    parser.add_argument('dataset')

    args = parser.parse_args()

    logging_setup(args.verbose, args.logfile)
    logging.getLogger("d3m.metadata.pipeline_run").setLevel(logging.ERROR)

    search(args)
