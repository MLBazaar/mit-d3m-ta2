import json
from collections import defaultdict
from datetime import timedelta
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from d3m.metadata.base import Context

from ta2.core import TA2Core, to_dicts


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


@patch('ta2.core.Pipeline.from_yaml')
@patch('ta2.core.os.makedirs')
def test_pipelinesearcher_defaults(makedirs_mock, from_yaml_mock):
    instance = TA2Core()

    expected_calls = [
        call('output/pipeline_runs', exist_ok=True),
        call('output/pipelines_ranked', exist_ok=True),
        call('output/pipelines_scored', exist_ok=True),
        call('output/pipelines_searched', exist_ok=True),
    ]
    assert makedirs_mock.call_args_list == expected_calls

    assert instance.input == 'input'
    assert instance.output == 'output'
    assert not instance.dump
    assert instance.ranked_dir == 'output/pipelines_ranked'
    assert instance.scored_dir == 'output/pipelines_scored'
    assert instance.searched_dir == 'output/pipelines_searched'
    assert instance.data_pipeline == from_yaml_mock.return_value
    assert instance.scoring_pipeline == from_yaml_mock.return_value


@patch('ta2.core.Pipeline.from_yaml')
@patch('ta2.core.os.makedirs')
def test_pipelinesearcher(makedirs_mock, from_yaml_mock):
    instance = TA2Core(input_dir='new-input', output_dir='new-output', dump=True)

    expected_calls = [
        call('new-output/pipeline_runs', exist_ok=True),
        call('new-output/pipelines_ranked', exist_ok=True),
        call('new-output/pipelines_scored', exist_ok=True),
        call('new-output/pipelines_searched', exist_ok=True),
    ]
    assert makedirs_mock.call_args_list == expected_calls

    assert instance.input == 'new-input'
    assert instance.output == 'new-output'
    assert instance.dump
    assert instance.ranked_dir == 'new-output/pipelines_ranked'
    assert instance.scored_dir == 'new-output/pipelines_scored'
    assert instance.searched_dir == 'new-output/pipelines_searched'
    assert instance.data_pipeline == from_yaml_mock.return_value
    assert instance.scoring_pipeline == from_yaml_mock.return_value


@pytest.mark.skip(reason="this needs to be fixed")
def test_pipelinesearcher_find_datasets(tmp_path):
    input_dir = tmp_path / 'test-input'
    input_dir.mkdir()

    content = {
        'about': {
            'datasetID': None
        }
    }

    num_datasets = 3
    for i in range(num_datasets):
        dataset_dir = input_dir / 'dataset-{}'.format(i)
        dataset_dir.mkdir()

        content['about']['datasetID'] = 'dataset-{}'.format(i)

        file = dataset_dir / 'datasetDoc.json'
        file.write_text(json.dumps(content))

    result = TA2Core._find_datasets(input_dir)

    assert len(result) == num_datasets

    for i in range(num_datasets):
        dataset_id = 'dataset-{}'.format(i)

        assert dataset_id in result
        assert result[dataset_id] == 'file://{}/{}/datasetDoc.json'.format(input_dir, dataset_id)


@pytest.mark.skip(reason="no way of currently testing this")
@patch('ta2.core.d3m_evaluate')
@patch('ta2.core.Pipeline.from_yaml', new=MagicMock())
def test_pipelinesearcher_score_pipeline(evaluate_mock):
    instance = TA2Core()
    expected_scores = [MagicMock(value=[1])]
    evaluate_mock.return_value = (expected_scores, expected_scores)

    # parameters
    dataset = {}
    problem = {'problem': {'performance_metrics': None}}
    pipeline_mock = MagicMock()
    metrics = {'test': 'metric'}
    random_seed = 0
    folds = 5
    stratified = False
    shuffle = False

    data_params = {
        'number_of_folds': json.dumps(folds),
        'stratified': json.dumps(stratified),
        'shuffle': json.dumps(shuffle),
    }

    # with custom metrics
    instance.score_pipeline(
        dataset, problem, pipeline_mock,
        metrics=metrics, random_seed=random_seed,
        folds=folds, stratified=stratified, shuffle=shuffle
    )

    evaluate_mock.assert_called_with(
        pipeline=pipeline_mock,
        inputs=[dataset],
        data_pipeline=instance.data_pipeline,
        scoring_pipeline=instance.scoring_pipeline,
        problem_description=problem,
        data_params=data_params,            # folds, stratified, shuffle
        metrics=metrics,                    # custom metrics
        context=Context.TESTING,
        random_seed=random_seed,
        data_random_seed=random_seed,
        scoring_random_seed=random_seed,
        volumes_dir=instance.static
    )

    assert pipeline_mock.cv_scores == [score.value[0] for score in expected_scores]

    # with problem metrics

    instance.score_pipeline(
        dataset, problem, pipeline_mock,
        metrics=None, random_seed=random_seed,
        folds=folds, stratified=stratified, shuffle=shuffle
    )

    evaluate_mock.assert_called_with(
        pipeline=pipeline_mock,
        inputs=[dataset],
        data_pipeline=instance.data_pipeline,
        scoring_pipeline=instance.scoring_pipeline,
        problem_description=problem,
        data_params=data_params,                            # folds, stratified, shuffle
        metrics=problem['problem']['performance_metrics'],  # custom metrics
        context=Context.TESTING,
        random_seed=random_seed,
        data_random_seed=random_seed,
        scoring_random_seed=random_seed,
        volumes_dir=instance.static
    )

    assert pipeline_mock.cv_scores == [score.value[0] for score in expected_scores]


@patch('ta2.core.datetime')
@patch('ta2.core.Pipeline.from_yaml', new=MagicMock())
def test_pipelinesearcher__check_stop(datetime_mock):
    datetime_mock.now = MagicMock(return_value=10)

    # no stop
    instance = TA2Core()
    instance._stop = False       # normally, set in `TA2Core._setup_search`
    instance.timeout = None      # normally, set in `TA2Core._setup_search`

    assert instance._check_stop() is None

    # stop by `_stop` attribute
    instance._stop = True

    with pytest.raises(KeyboardInterrupt):
        instance._check_stop()

    # stop by `max_end_time`
    instance._stop = False
    instance.timeout = 10
    instance.max_end_time = 5

    with pytest.raises(KeyboardInterrupt):
        instance._check_stop()


@patch('ta2.core.Pipeline.from_yaml', new=MagicMock())
def test_pipelinesearcher_stop():
    instance = TA2Core()

    assert not hasattr(instance, '_stop')

    # setting _stop
    instance.stop()
    assert instance._stop


@patch('ta2.core.Pipeline.from_yaml', new=MagicMock())
def test_pipelinesearcher__setup_search():
    instance = TA2Core()

    assert hasattr(instance, 'solutions')
    assert not hasattr(instance, '_stop')
    assert not hasattr(instance, 'done')
    assert not hasattr(instance, 'start_time')
    assert not hasattr(instance, 'timeout')
    assert not hasattr(instance, 'max_end_time')

    # without timeout
    instance.timeout = None
    instance._setup_search(None)

    assert instance.solutions == []
    assert instance._stop is False
    assert instance.done is False
    assert hasattr(instance, 'start_time')
    assert instance.timeout is None
    assert instance.max_end_time is None

    # with timeout
    instance._setup_search(0.5)

    assert instance.timeout == 0.5
    assert instance.max_end_time == instance.start_time + timedelta(seconds=0.5)
