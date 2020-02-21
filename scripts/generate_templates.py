import argparse
import glob
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import Problem
from d3m.utils import yaml_load_all

LOGGER = logging.getLogger(__name__)
TUNING_PARAMETER = 'https://metadata.datadrivendiscovery.org/types/TuningParameter'


def load_pipeline(pipeline):
    with open(pipeline) as _pipeline:
        if pipeline.endswith('.json'):
            pipeline = Pipeline.from_json(_pipeline)
        else:
            pipeline = Pipeline.from_yaml(_pipeline)

    return pipeline


def get_default_step_hyperparams(step):
    default_tunable_hyperparams = {}
    for name, hp in step.get_all_hyperparams().items():
        if TUNING_PARAMETER not in hp.semantic_types:
            continue

        default_tunable_hyperparams[name] = hp.get_default()

    return default_tunable_hyperparams


def clean_hyperparams(pipeline):
    for step in pipeline.steps:
        default_tunable_hyperparams = get_default_step_hyperparams(step)

        for name, value in step.hyperparams.items():
            if name in default_tunable_hyperparams.keys():
                value['data'] = default_tunable_hyperparams[name]

    return pipeline


def pipeline_to_template(pipeline_path):
    pipeline = load_pipeline(pipeline_path)
    template = clean_hyperparams(pipeline)

    template.id = ''
    template.schema = 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json'
    template.created = datetime(2016, 11, 11, 12, 30, tzinfo=timezone.utc)

    return template


def write_template(templates_path, template):
    template_id = template.get_digest()[:12]
    template_path = os.path.join(templates_path, template_id + '.json')

    with open(template_path, 'w') as template_file:
        print("Creating template {}".format(template_path))
        template.to_json(template_file)


def generate_templates(pipelines_path, templates_path):
    for pipeline in os.listdir(pipelines_path):
        pipeline_path = os.path.join(pipelines_path, pipeline)
        try:
            template = pipeline_to_template(pipeline_path)
            write_template(templates_path, template)
        except Exception as ex:
            print(ex)


def read_pipeline_run(pipeline_run_path):
    data = open(pipeline_run_path)
    docs = yaml_load_all(stream=data)
    res = []
    for doc in docs:
        res.append(doc)

    data.close()

    return res


def load_problem(root_path, phase):
    path = os.path.join(root_path, phase, 'problem_' + phase, 'problemDoc.json')
    return Problem.load(problem_uri=path)


def detect_data_modality(dataset_doc):
    with open(dataset_doc) as f:
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


def get_dataset_info(dataset_name, datasets_path):

    dataset_root = os.path.join(datasets_path, dataset_name)

    if not os.path.exists(dataset_root):
        dataset_root += '_MIN_METADATA'

    dataset_doc = os.path.join(dataset_root, 'TRAIN', 'dataset_TRAIN', 'datasetDoc.json')
    dataset_root = 'file://' + os.path.abspath(dataset_root)
    problem = load_problem(dataset_root, 'TRAIN')

    # Dataset Meta
    data_modality = detect_data_modality(dataset_doc)
    task_type = problem['problem']['task_keywords'][0].name.lower()
    task_subtype = problem['problem']['task_keywords'][1].name.lower()

    return data_modality, task_type, task_subtype


def get_template_id(pipeline_id, pipelines_path, templates_path):

    pipeline_path = os.path.join(pipelines_path, '{}.json'.format(pipeline_id))
    if not os.path.isfile(pipeline_path):
        raise ValueError('Can not find: {}'.format(pipeline_path))

    template = pipeline_to_template(pipeline_path)
    write_template(templates_path, template)
    return template.get_digest()[:12]


def produce_phase(pipeline_run):
    """Produce result with Produce phase data."""
    scores = pipeline_run['run']['results']['scores']

    if len(scores) > 1:
        raise ValueError('This run has more than one score!')

    scores = scores[0]

    return {
        'metric': scores['metric']['metric'],
        'context': pipeline_run['context'],
        'normalized_score': scores['normalized']
    }


def extract_pipeline_run(pipeline_run, pipelines_path, templates_path, datasets_path):
    dataset_id = pipeline_run['datasets'][0]['id']
    phase = pipeline_run['run']['phase']
    succeed = pipeline_run.get('status').get('state')
    pipeline_id = pipeline_run['pipeline']['id']

    if dataset_id.endswith('TRAIN'):
        dataset_name = dataset_id.replace('_dataset_TRAIN', '')
    else:
        dataset_name = dataset_id.replace('_dataset_SCORE', '')

    # TODO: Lazy Loader
    data_modality, task_type, task_subtype = get_dataset_info(dataset_name, datasets_path)

    template_id = get_template_id(pipeline_id, pipelines_path, templates_path)

    result = {
        'dataset': dataset_name,
        'pipeline_id': pipeline_id,
        'template_id': template_id,
        'modality': data_modality,
        'type': task_type,
        'subtype': task_subtype,
        'phase': phase,
        'succeed': succeed,
    }

    if phase == 'PRODUCE' and succeed != 'FAILURE':
        try:
            score = produce_phase(pipeline_run)
            result.update(score)
        except:
            # Timeout
            result['phase'] = 'TIMEOUT'

    return result, succeed


def extract_meta_information(pipeline_runs, pipelines_path, templates_path, datasets_path):
    pipeline_runs_path = os.path.join(pipeline_runs, '*')

    results = []
    errored = []
    discarded = []

    for pipeline_run_path in glob.glob(pipeline_runs_path):
        pipeline_runs = load_pipeline_run(pipeline_run_path)

        data_extracted = []

        failed = False

        for pipeline_run in pipeline_runs:
            try:
                run_data, run_status = extract_pipeline_run(
                    pipeline_run, pipelines_path, templates_path, datasets_path)

                failed = run_status == 'FAILURE'

                data_extracted.append(run_data)

            except Exception as e:
                LOGGER.warning('Failed %s with: %s', pipeline_run_path, e)
                continue

        if not failed:
            results.extend(data_extracted)

        else:
            LOGGER.warning('Pipeline run %s discarded.', pipeline_run_path)
            discarded.append(data_extracted)

    return results, discarded


def apply_mean_score(df):
    mean_score = df.groupby(['pipeline_id', 'context'])['normalized_score'].mean()
    mean_score = mean_score.reset_index()
    mean_score.rename(columns={'normalized_score': 'mean_score'}, inplace=True)
    return df.merge(mean_score, on=['pipeline_id', 'context'], how='left')


def z_score(x):
    if len(x) == 1 or x.std() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)

    return (x - x.mean()) / x.std()


def apply_z_score(df):
    z_scores = df.groupby('dataset').normalized_score.apply(z_score)
    df['z_score'] = z_scores
    templates_z_score = df.groupby('template_id').z_score.mean()
    del df['z_score']

    return df.merge(templates_z_score, how='left', left_on='template_id', right_index=True)


def generate_metadata_report(pipeline_runs, pipelines_path, templates_path, datasets_path, report):

    results, discarded = extract_meta_information(
        pipeline_runs, pipelines_path, templates_path, datasets_path)

    if report is None:
        report = os.path.join(templates_path, 'templates.csv')

    df = pd.DataFrame(results)
    df = apply_mean_score(df)
    df = apply_z_score(df)
    df.to_csv(report, index=False)

    if errored:
        with open('errors.txt', 'w') as f:
            for error in errored:
                f.write('{}\n'.format(error))


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate new templates from pipeline runs and the metadata reffered to them.')
    parser.add_argument('pipeline_runs_path', help='Path to the pipeline runs folder')
    parser.add_argument('pipelines_path', help='Path to the pipelines folder')
    parser.add_argument('templates_path', help='Path to the templates folder')
    parser.add_argument('datasets_path', help='Path where the datasets are located')
    parser.add_argument('-r', '--report', help='Path to the CSV file where scores will be dumped.')

    return parser.parse_args()


def main():
    args = parse_args()
    generate_metadata_report(
        args.pipeline_runs_path,
        args.pipelines_scored_path,
        args.templates_path,
        args.datasets_path,
        args.report,
    )


if __name__ == '__main__':
    main()
