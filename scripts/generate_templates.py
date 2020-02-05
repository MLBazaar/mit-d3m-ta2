import os
import traceback
import sys

import pandas as pd

from d3m.metadata.pipeline import Pipeline


def load_pipeline(pipeline):
    with open(pipeline) as _pipeline:
        if pipeline.endswith('.json'):
            pipeline = Pipeline.from_json(_pipeline)
        else:
            pipeline = Pipeline.from_yaml(_pipeline)

    pipeline.id = pipeline.get_digest()

    return pipeline


def generate_new_templates(path):
    errors = []
    for pipeline in os.listdir(path):
        pipeline_path = os.path.join(path, pipeline)
        try:
            pipeline = load_pipeline(pipeline_path)
            new_path = os.path.join('templates', pipeline.id + '.json')

            with open(new_path, 'w') as pipeline_file:
                pipeline.to_json(pipeline_file)

            print('OK:', pipeline.id)

        except Exception:
            print('ERROR:', pipeline)
            traceback.print_exc(file=sys.stdout)
            errors.append(pipeline_path)

    if errors:
        with open('errors.txt', 'w') as f:
            for error in errors:
                print(error, file=f)


if __name__ == '__main__':

    # Make all the d3m traces quiet
    null = open(os.devnull, 'w')
    sys.stderr = null

    path = sys.argv[1]
    generate_new_templates(path)
