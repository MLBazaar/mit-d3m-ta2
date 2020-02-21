import json
import logging
import os
import warnings

import yaml
from btb.tuning.hyperparams.boolean import BooleanHyperParam
from btb.tuning.hyperparams.categorical import CategoricalHyperParam
from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam
from btb.tuning.tunable import Tunable
from d3m.metadata.hyperparams import Bounded, Enumeration, Uniform, UniformBool, UniformInt
from d3m.metadata.pipeline import Pipeline

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

TUNING_PARAMETER = 'https://metadata.datadrivendiscovery.org/types/TuningParameter'

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# DATA_AUGMENTATION = 'd3m.primitives.data_augmentation.datamart_augmentation.Common'


def get_default_pipeline_parameters(json_pipeline, tunables):
    """Returns default values from the json pipeline file."""
    steps = json_pipeline['steps']
    default_hyperparams = {}

    for step in range(len(steps)):
        step_hyperparams = steps[step].get('hyperparams')

        if step_hyperparams is None:
            continue

        for name, hyperparam in step_hyperparams.items():
            default_hyperparams[(str(step), name)] = hyperparam['data']

    for name, tunable in tunables.items():
        if default_hyperparams.get(name) is None:
            default_hyperparams[name] = tunable.default

    return default_hyperparams


def _generate_bounded_hyperparam(hyperparam, default):
    hp_dict = {
        'min': hyperparam.lower,
        'max': hyperparam.upper,
        'default': default,
        'include_min': hyperparam.lower_inclusive,
        'include_max': hyperparam.upper_inclusive,
    }

    if hyperparam.structural_type is float:
        btb_hyperparam = FloatHyperParam(**hp_dict)

    elif hyperparam.structural_type is int:
        btb_hyperparam = IntHyperParam(**hp_dict)

    else:
        return None

    return btb_hyperparam


def extract_tunable_hyperparams(pipeline):

    # state tunable hyperparameters for this pipeline
    tunable_hyperparams = {}

    # obtain all the hyperparameters
    hyperparams = pipeline.get_all_hyperparams()

    for step_num, (step_hyperparams, step) in enumerate(zip(hyperparams, pipeline.steps)):

        for name, hyperparam in step_hyperparams.items():
            # all logic goes here
            if TUNING_PARAMETER not in hyperparam.semantic_types:
                continue

            btb_hyperparam = None
            default = step.hyperparams.get(name, {}).get('data', hyperparam.get_default())

            if isinstance(hyperparam, Bounded):
                btb_hyperparam = _generate_bounded_hyperparam(hyperparam, default)

            elif isinstance(hyperparam, Enumeration):
                btb_hyperparam = CategoricalHyperParam(
                    choices=hyperparam.values,
                    default=default
                )

            elif isinstance(hyperparam, UniformBool):
                btb_hyperparam = BooleanHyperParam(default=default)

            elif isinstance(hyperparam, (UniformInt, Uniform)):
                btb_hyperparam = _generate_bounded_hyperparam(hyperparam, default)

            if btb_hyperparam is not None:
                tunable_hyperparams[(str(step_num), hyperparam.name)] = btb_hyperparam

    return tunable_hyperparams


def load_pipeline(path, tunables=True, defaults=True):
    """Load a d3m json or yaml pipeline."""

    if not os.path.exists(path):
        base_path = os.path.abspath(os.path.dirname(__file__))

        path = os.path.join('templates', path)
        path = os.path.join(base_path, path)

    if not os.path.isfile(path):
        raise ValueError('Could not find pipeline: {}'.format(path))

    LOGGER.warn('Loading pipeline from %s', path)
    with open(path) as pipeline:
        if path.endswith('yml'):
            data = yaml.safe_load(pipeline)

        else:
            data = json.load(pipeline)

    pipeline = Pipeline.from_json_structure(data)

    if tunables:
        # extract tunable hyperparameters
        tunable_hyperparameters = extract_tunable_hyperparams(pipeline)

        return pipeline, tunable_hyperparameters

    return pipeline


class LazyLoader(dict):
    def __init__(self, keys, templates_dir):
        super().__init__({key: None for key in keys})
        self._templates_dir = templates_dir

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if value is not None:
            return value

        path = os.path.join(self._templates_dir, key)
        if not path.endswith('.json'):
            path += '.json'

        template, tunable_hp = load_pipeline(path)
        self[key] = template

        return Tunable(tunable_hp)
