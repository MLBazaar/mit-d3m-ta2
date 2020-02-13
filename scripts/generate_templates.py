import argparse
import os
import sys
import traceback
from datetime import datetime, timezone

from d3m.metadata.pipeline import Pipeline

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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate templates from pipelines')
    parser.add_argument('pipelines_path', help='Path to the pipelines folder')
    parser.add_argument('templates_path', help='Path to the templates folder')

    return parser.parse_args()


def main():
    args = parse_args()
    generate_templates(args.pipelines_path, args.templates_path)


if __name__ == '__main__':
    main()
