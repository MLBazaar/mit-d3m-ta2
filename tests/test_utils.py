from collections import defaultdict

import numpy as np

from ta2.utils import to_dicts


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

    assert set(result) == set(expected_hyperparams)
