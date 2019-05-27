from collections import defaultdict
from unittest.mock import patch

import numpy as np
from d3m.metadata.pipeline import Pipeline

from ta2.search import PipelineSearcher, to_dicts


def test_to_dicts():
    hyperparams = {
        # (block, hyperparameter): value
        ('block-1', 'param-1'): np.int_(1),                    # np.integer
        ('block-2', 'param-1'): np.float32(1.0),               # np.floating
        ('block-2', 'param-2'): np.arange(3, dtype=np.uint8),  # np.ndarray
        ('block-3', 'param-1'): np.bool_(True),                # np.bool_
        ('block-3', 'param-2'): 'None',                        # None
        ('block-3', 'param-3'): 1,
        ('block-4', 'param-1'): 1.0,
        ('block-4', 'param-2'): [1, 2, 3],
        ('block-4', 'param-3'): True,
        ('block-4', 'param-4'): None,
        ('block-5', 'param-4'): 'other value'
    }

    result = to_dicts(hyperparams)

    expected_hyperparams = defaultdict(dict)
    for (block, hyperparameter), value in hyperparams.items():
        expected_hyperparams[block][hyperparameter] = value

    expected_hyperparams['block-1']['param-1'] = 1
    expected_hyperparams['block-2']['param-1'] = 1.0
    expected_hyperparams['block-2']['param-2'] = [0, 1, 2]
    expected_hyperparams['block-3']['param-1'] = True
    expected_hyperparams['block-3']['param-2'] = None

    assert result == expected_hyperparams


@patch('ta2.search.os.makedirs')
def test_pipelinesearcher(makedirs_mock):
    # static methods
    assert hasattr(PipelineSearcher, '_find_datasets')
    assert hasattr(PipelineSearcher, '_new_pipeline')

    # default parameters
    instance = PipelineSearcher()

    makedirs_mock.assert_called_with(instance.ranked_dir, exist_ok=True)

    assert instance.input == 'input'
    assert instance.output == 'output'
    assert instance.dump
    assert instance.ranked_dir == '{}/pipelines_ranked'.format(instance.output)
    assert isinstance(instance.data_pipeline, Pipeline)
    assert isinstance(instance.scoring_pipeline, Pipeline)

    # other parameters
    instance = PipelineSearcher(input_dir='new-input', output_dir='new-output', dump=False)

    makedirs_mock.assert_called_with(instance.ranked_dir, exist_ok=True)

    assert instance.input == 'new-input'
    assert instance.output == 'new-output'
    assert not instance.dump
    assert instance.ranked_dir == '{}/pipelines_ranked'.format(instance.output)
    assert isinstance(instance.data_pipeline, Pipeline)
    assert isinstance(instance.scoring_pipeline, Pipeline)
    assert instance.datasets == {}
