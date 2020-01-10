import argparse
import importlib
import logging
import os
import warnings
from collections import defaultdict

import yaml
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.hyperparams import Union
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

TUNING_PARAMETER = 'https://metadata.datadrivendiscovery.org/types/TuningParameter'

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_AUGMENTATION = 'd3m.primitives.data_augmentation.datamart_augmentation.Common'


def extract_pipeline_tunables(pipeline):
    tunable_hyperparameters = defaultdict(dict)
    for step, step_hyperparams in enumerate(pipeline.get_free_hyperparams()):
        for name, hyperparam in step_hyperparams.items():
            if TUNING_PARAMETER not in hyperparam.semantic_types:
                continue

            if isinstance(hyperparam, Union):
                hyperparam = hyperparam.default_hyperparameter

            try:
                param_type = hyperparam.structural_type.__name__
                param_type = 'string' if param_type == 'str' else param_type
                if param_type == 'bool':
                    param_range = [True, False]
                elif hasattr(hyperparam, 'values'):
                    param_range = hyperparam.values
                else:
                    lower = hyperparam.lower
                    upper = hyperparam.upper
                    if upper is None:
                        upper = lower + 1000
                    elif upper > lower:
                        if param_type == 'int':
                            upper = upper - 1
                        elif param_type == 'float':
                            upper = upper - 0.0001

                    param_range = [lower, upper]

            except AttributeError:
                LOGGER.warn('Warning! skipping: %s, %s, %s', step, name, hyperparam)
                continue

            try:
                # If the line above did not crash, we are safe
                tunable_hyperparameters[step][name] = {
                    'type': param_type,
                    'range': param_range,
                    'default': hyperparam.get_default()
                }

            except OverflowError:
                LOGGER.warn('Warning! Overflow: %s, %s, %s', step, name, hyperparam)
                continue

    return tunable_hyperparameters


def get_tunable_hyperparameters(tunables, defaults):
    tunable_hyperparameters = defaultdict(dict)
    for key, tunable in tunables:
        step, param_name = key
        param_type = tunable.param_type.name.lower()
        param_range = tunable._param_range
        tunable_hyperparameters[step][param_name] = {
            'type': param_type,
            'range': param_range,
            'default': defaults[key]
        }

    return dict(tunable_hyperparameters)


def load_template(template_name, data_augmentation=None):
    """load a simplified version of a yaml pipeline, with hyperparameters."""

    if os.path.exists(template_name):
        template_path = template_name
    else:
        template_path = os.path.join(TEMPLATES_DIR, template_name)

    LOGGER.info('Loading template %s', template_path)
    with open(template_path, 'r') as template_file:
        template = yaml.safe_load(template_file)

    steps = template['steps']

    pipeline = Pipeline()
    pipeline.add_input(name='inputs')

    if data_augmentation:
        LOGGER.info('Adding Data Augmentation Step')
        step_0 = {
            'primitive': DATA_AUGMENTATION,
            'hyperparams': {
                'system_identifier': {
                    'data': 'NYU'
                },
                'search_result': {
                    'data': data_augmentation
                }
            }
        }
        steps.insert(0, step_0)

    for step_num, primitive_config in enumerate(steps):
        primitive_name = primitive_config['primitive']
        LOGGER.info("Loading primitive {}".format(primitive_name))
        primitive = index.get_primitive(primitive_name)
        step = PrimitiveStep(primitive=primitive)

        if step_num == 0:
            data_reference = 'inputs.0'
        else:
            data_reference = 'steps.{}.produce'.format(step_num - 1)

        arguments = primitive_config.get('arguments', dict())
        if data_augmentation and arguments:
            for argument in arguments.values():
                data = argument.get('data')
                if data and 'step' in data:
                    data = data.split('.')
                    data[1] = str(int(data[1]) + 1)
                    argument['data'] = '.'.join(data)

        if 'inputs' not in arguments:
            arguments['inputs'] = {
                'data': data_reference,
            }

        for name, argument in arguments.items():
            step.add_argument(
                name=name,
                argument_type=ArgumentType[argument.get('type', 'CONTAINER')],
                data_reference=argument['data']
            )

        hyperparams = primitive_config.get('hyperparams', dict())
        for name, hyperparam in hyperparams.items():
            data = hyperparam['data']
            if isinstance(data, dict):
                data = data['value']
                if isinstance(data, dict):
                    if 'class' in data:
                        package, class_name = data['class'].rsplit('.', 1)
                        data = getattr(importlib.import_module(package), class_name)
                    else:
                        raise ValueError('Unsupported template')

            step.add_hyperparameter(
                name=name,
                argument_type=ArgumentType[hyperparam.get('type', 'VALUE')],
                data=data
            )

        outputs = primitive_config.get('outputs', ('produce',))
        for output in outputs:
            step.add_output(output)

        pipeline.add_step(step)

    data_reference = 'steps.{}.produce'.format(len(steps) - 1)
    pipeline.add_output(name='output predictions', data_reference=data_reference)

    if 'tunable_hyperparameters' in template:
        LOGGER.info('Using predefined tunable hyperparameters')
        tunable_hyperparameters = template['tunable_hyperparameters']
    else:
        LOGGER.info('Extracting tunables from pipeline')
        tunable_hyperparameters = extract_pipeline_tunables(pipeline)

    if data_augmentation:
        tunable_hyperparameters = {
            str(int(key) + 1): value
            for key, value in tunable_hyperparameters.items()
        }

    LOGGER.info("Pipeline %s loaded", pipeline.id)
    return pipeline, tunable_hyperparameters


def add_tunable_hyperparameters(input_path, output_path):
    _, tunables, defaults = load_template(input_path)
    tunable_hyperparameters = get_tunable_hyperparameters(tunables, defaults)

    with open(input_path, 'r') as template_file:
        template = yaml.safe_load(template_file)

    template['tunable_hyperparameters'] = tunable_hyperparameters

    if output_path:
        with open(output_path, 'w') as template_file:
            yaml.safe_dump(template, template_file)

    else:
        print(yaml.safe_dump(template))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage templates')
    parser.add_argument('-o', '--output')
    parser.add_argument('input', help='Name or path to the template')

    args = parser.parse_args()

    add_tunable_hyperparameters(args.input, args.output)
