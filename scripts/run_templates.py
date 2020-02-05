import argparse
import io
import json
import os
import sys
import tarfile
import urllib
from multiprocessing import Manager, Process

import pandas as pd

from d3m import index
from d3m.container.dataset import Dataset
from d3m.metadata.base import Context
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import parse_problem_description
from d3m.runtime import DEFAULT_SCORING_PIPELINE_PATH, Runtime, score


from ta2.__main__ import load_problem, score_pipeline


def detect_data_modality(doc_path):
    with open(doc_path) as f:
        dataset_doc = json.load(f)

    resources = list()
    for resource in dataset_doc['dataResources']:
        resources.append(resource['resType'])

    if len(resources) == 1:
        return 'single_table'
    else:
        for res in resources:
            if res == 'edgeList':
                return 'graph'
            elif res not in ('table', 'raw'):
                return res

    return 'multi_table'


def run_pipeline(output, dataset_root, problem, pipeline_path, static=None):
    score = score_pipeline(dataset_root, problem, pipeline_path, static)
    output.append(score)


def test_pipelines(pipelines, dataset_root):
    datasets_path, dataset_name = dataset_root.rsplit('/', 1)
    # train_dataset = load_dataset(dataset_root, 'TRAIN')
    # test_dataset = load_dataset(dataset_root, 'TEST')
    problem = load_problem(dataset_root, 'TRAIN')
    evaluated_pipelines = list()

    task_type = problem['problem']['task_keywords'][0].name.lower()
    task_subtype = problem['problem']['task_keywords'][1].name.lower()

    path = os.path.join(dataset_root, 'TRAIN', 'dataset_' + 'TRAIN', 'datasetDoc.json')
    data_modality = detect_data_modality(path)

    for pipeline in os.listdir(pipelines):
        print(pipeline.rsplit('.', 1)[0] + ':', end=' ')

        pipeline_path = os.path.join(pipelines, pipeline)
        try:
            with Manager() as manager:
                output = manager.list()
                process = Process(
                    target=run_pipeline,
                    args=(output, dataset_root, problem, pipeline_path)
                )
                process.daemon = True
                process.start()
                process.join(10)
                process.terminate()

                if output:
                    print("success -", output[0])
                    evaluated_pipelines.append({
                        'dataset_name': dataset_name,
                        'template_id': pipeline,
                        'problem_type': '{}_{}'.format(data_modality, task_type),
                        'task_type': task_type,
                        'task_subtype': task_subtype,
                    })

                elif process.exitcode == -15:
                    print("timeout")
                else:
                    print("error")

        except Exception as ex:
            print("error")

    df = pd.DataFrame(evaluated_pipelines)
    df.to_csv('{}_{}.csv'.format(data_modality, task_type), index=False)


if __name__ == '__main__':

    # Make all the d3m traces quiet
    null = open(os.devnull, 'w')
    sys.stderr = null

    pipelines, dataset = sys.argv[1:3]
    test_pipelines(pipelines, dataset)
