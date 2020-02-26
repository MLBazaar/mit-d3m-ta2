from datetime import timedelta
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import pytest

from ta2.core import ScoringError, TA2Core


class TestTA2Core(TestCase):

    @patch('ta2.core.Pipeline.from_yaml')
    @patch('ta2.core.os.makedirs')
    def test_pipelinesearcher_defaults(self, makedirs_mock, from_yaml_mock):
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
        assert instance.static == 'static'
        assert instance.folds == 5
        assert instance.max_errors == 5
        assert not instance.dump
        assert not instance.hard_timeout
        assert not instance.ignore_errors
        assert not instance.subprocess_timeout
        assert not instance.store_summary

        assert instance.runs_dir == 'output/pipeline_runs'
        assert instance.ranked_dir == 'output/pipelines_ranked'
        assert instance.scored_dir == 'output/pipelines_scored'
        assert instance.searched_dir == 'output/pipelines_searched'

        assert instance.data_pipeline == from_yaml_mock.return_value
        assert instance.scoring_pipeline == from_yaml_mock.return_value

    @patch('ta2.core.Pipeline.from_yaml')
    @patch('ta2.core.os.makedirs')
    def test_pipelinesearcher(self, makedirs_mock, from_yaml_mock):
        instance = TA2Core(input_dir='new-input', output_dir='new-output', static_dir='new-static',
                           dump=True, hard_timeout=True, ignore_errors=True, cv_folds=7,
                           subprocess_timeout=720, max_errors=7, store_summary=True)

        expected_calls = [
            call('new-output/pipeline_runs', exist_ok=True),
            call('new-output/pipelines_ranked', exist_ok=True),
            call('new-output/pipelines_scored', exist_ok=True),
            call('new-output/pipelines_searched', exist_ok=True),
        ]
        assert makedirs_mock.call_args_list == expected_calls

        assert instance.input == 'new-input'
        assert instance.output == 'new-output'
        assert instance.static == 'new-static'
        assert instance.folds == 7
        assert instance.max_errors == 7
        assert instance.subprocess_timeout == 720

        assert instance.dump
        assert instance.hard_timeout
        assert instance.ignore_errors
        assert instance.store_summary

        assert instance.runs_dir == 'new-output/pipeline_runs'
        assert instance.ranked_dir == 'new-output/pipelines_ranked'
        assert instance.scored_dir == 'new-output/pipelines_scored'
        assert instance.searched_dir == 'new-output/pipelines_searched'

        assert instance.data_pipeline == from_yaml_mock.return_value
        assert instance.scoring_pipeline == from_yaml_mock.return_value

    @patch('ta2.core.TA2Core.subprocess_evaluate')
    @patch('ta2.core.Pipeline.from_yaml', new=MagicMock())
    def test_pipelinesearcher_score_pipeline_failed(self, mock_subprocess):
        mock_subprocess.return_value = (False, [MagicMock()])
        instance = TA2Core()

        with pytest.raises(ScoringError):
            instance.score_pipeline({}, {'problem': {'performance_metrics': None}}, MagicMock())

    @patch('yaml.dump_all')
    @patch('builtins.open')
    @patch('ta2.core.TA2Core.subprocess_evaluate')
    @patch('ta2.core.Pipeline.from_yaml', new=MagicMock())
    def test_score_pipeline_dump_summary(self, mock_subprocess, mock_file, mock_yaml):
        all_results = [MagicMock()]
        mock_subprocess.return_value = ([MagicMock()], all_results)
        instance = TA2Core(store_summary=True)

        pipeline = MagicMock()
        mock_file.reset_mock()

        instance.score_pipeline({}, {'problem': {'performance_metrics': MagicMock()}}, pipeline)

        runs = [res.pipeline_run.to_json_structure() for res in all_results]

        mock_file.assert_called_once_with('output/pipeline_runs/{}.yml'.format(pipeline.id), 'w')
        mock_yaml.assert_called_once_with(runs, mock_file().__enter__(), default_flow_style=False)


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
