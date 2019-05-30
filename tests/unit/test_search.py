import json
from collections import defaultdict
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from d3m.metadata.base import Context
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import TaskType

from ta2.search import PIPELINES_DIR, PipelineSearcher, to_dicts


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

    result = PipelineSearcher._find_datasets(input_dir)

    assert len(result) == num_datasets

    for i in range(num_datasets):
        dataset_id = 'dataset-{}'.format(i)

        assert dataset_id in result
        assert result[dataset_id] == 'file://{}/{}/datasetDoc.json'.format(input_dir, dataset_id)


@patch('ta2.search.Pipeline.from_yaml')
@patch('ta2.search.Pipeline.from_json')
def test_pipelinesearcher_load_pipeline(json_loader_mock, yaml_loader_mock):
    instance = PipelineSearcher()
    open_mock = mock_open(read_data='data')

    json_loader_mock.reset_mock()
    yaml_loader_mock.reset_mock()

    # yaml file
    with patch('ta2.search.open', open_mock) as _:
        instance._load_pipeline('test.yml')

    open_mock.assert_called_with('{}/test.yml'.format(PIPELINES_DIR), 'r')

    assert yaml_loader_mock.call_count == 1
    assert json_loader_mock.call_count == 0

    # json file
    with patch('ta2.search.open', open_mock) as _:
        instance._load_pipeline('test.json')

    open_mock.assert_called_with('{}/test.json'.format(PIPELINES_DIR), 'r')

    assert yaml_loader_mock.call_count == 1
    assert json_loader_mock.call_count == 1

    # without file extension
    with patch('ta2.search.open', open_mock) as _:
        instance._load_pipeline('test')

    open_mock.assert_called_with('{}/test.json'.format(PIPELINES_DIR), 'r')

    assert yaml_loader_mock.call_count == 1
    assert json_loader_mock.call_count == 2


@patch('ta2.search.LOGGER.info')
def test_pipelinesearcher_get_template(logger_mock):
    instance = PipelineSearcher()
    data = {
        'problem': {
            'task_type': None
        }
    }

    # classification
    data['problem']['task_type'] = TaskType.CLASSIFICATION

    result = instance._get_template(None, data)  # dataset (None) is not used

    assert logger_mock.call_count == 1
    assert result == 'gradient_boosting_classification.all_hp.yml'

    # regression
    data['problem']['task_type'] = TaskType.REGRESSION

    result = instance._get_template(None, data)  # dataset (None) is not used

    assert logger_mock.call_count == 2
    assert result == 'gradient_boosting_regression.all_hp.yml'

    # not supported
    data['problem']['task_type'] = 'other-task-type'

    with pytest.raises(ValueError):
        instance._get_template(None, data)  # dataset (None) is not used


@patch('ta2.search.evaluate')
def test_pipelinesearcher_score_pipeline(evaluate_mock):
    instance = PipelineSearcher()
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
    result = instance.score_pipeline(
        dataset, problem, pipeline_mock,
        metrics=metrics, random_seed=random_seed,
        folds=folds, stratified=stratified, shuffle=shuffle
    )

    evaluate_mock.assert_called_with(
        pipeline_mock,
        instance.data_pipeline,
        instance.scoring_pipeline,
        problem,
        [dataset],
        data_params,            # folds, stratified, shuffle
        metrics,                # custom metrics
        context=Context.TESTING,
        random_seed=random_seed,
        data_random_seed=random_seed,
        scoring_random_seed=random_seed,
    )

    assert pipeline_mock.cv_scores == [score.value[0] for score in expected_scores]
    assert result == np.mean(pipeline_mock.cv_scores)

    # with problem metrics

    result = instance.score_pipeline(
        dataset, problem, pipeline_mock,
        metrics=None, random_seed=random_seed,
        folds=folds, stratified=stratified, shuffle=shuffle
    )

    evaluate_mock.assert_called_with(
        pipeline_mock,
        instance.data_pipeline,
        instance.scoring_pipeline,
        problem,
        [dataset],
        data_params,                               # folds, stratified, shuffle
        problem['problem']['performance_metrics'],  # problem metrics
        context=Context.TESTING,
        random_seed=random_seed,
        data_random_seed=random_seed,
        scoring_random_seed=random_seed,
    )

    assert pipeline_mock.cv_scores == [score.value[0] for score in expected_scores]
    assert result == np.mean(pipeline_mock.cv_scores)
