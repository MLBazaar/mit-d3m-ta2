from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from ta2.core import DATA_PIPELINE_PATH, DEFAULT_SCORING_PIPELINE_PATH, ScoringError, TA2Core


class TestTA2Core(TestCase):

    @patch('ta2.core.load_pipeline')
    @patch('ta2.core.os.makedirs')
    def test___init__defaults(self, makedirs_mock, load_pipeline_mock):
        # setup
        load_pipeline_mock.side_effect = ['data_pipeline', 'default_scoring_pipeline']

        # run
        instance = TA2Core()

        # assert
        expected_makedirs_calls = [
            call('output/pipeline_runs', exist_ok=True),
            call('output/pipelines_ranked', exist_ok=True),
            call('output/pipelines_scored', exist_ok=True),
            call('output/pipelines_searched', exist_ok=True),
        ]

        expceted_load_pipeline_calls = [
            call(DATA_PIPELINE_PATH),
            call(DEFAULT_SCORING_PIPELINE_PATH),
        ]

        assert makedirs_mock.call_args_list == expected_makedirs_calls

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

        assert instance.data_pipeline == 'data_pipeline'
        assert instance.scoring_pipeline == 'default_scoring_pipeline'

        assert load_pipeline_mock.call_args_list == expceted_load_pipeline_calls

    @patch('ta2.core.load_pipeline')
    @patch('ta2.core.os.makedirs')
    def test___init__not_defaults(self, makedirs_mock, load_pipeline_mock):
        # setup
        load_pipeline_mock.side_effect = ['data_pipeline', 'default_scoring_pipeline']

        # run
        instance = TA2Core(input_dir='new-input', output_dir='new-output', static_dir='new-static',
                           dump=True, hard_timeout=True, ignore_errors=True, cv_folds=7,
                           subprocess_timeout=720, max_errors=7, store_summary=True)

        # assert
        expected_makedirs_calls = [
            call('new-output/pipeline_runs', exist_ok=True),
            call('new-output/pipelines_ranked', exist_ok=True),
            call('new-output/pipelines_scored', exist_ok=True),
            call('new-output/pipelines_searched', exist_ok=True),
        ]

        expceted_load_pipeline_calls = [
            call(DATA_PIPELINE_PATH),
            call(DEFAULT_SCORING_PIPELINE_PATH),
        ]

        assert makedirs_mock.call_args_list == expected_makedirs_calls

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

        assert load_pipeline_mock.call_args_list == expceted_load_pipeline_calls

    def test_score_pipeline_failed(self):
        instance = MagicMock(autospec=TA2Core)
        instance.folds = 5
        instance.subprocess_evaluate.return_value = (False, [MagicMock()])

        with pytest.raises(ScoringError):
            TA2Core.score_pipeline(
                instance, {}, {'problem': {'performance_metrics': None}}, MagicMock()
            )

    @patch('yaml.dump_all')
    @patch('builtins.open')
    def test_score_pipeline_dump_summary(self, mock_file, mock_yaml):
        all_results = [MagicMock()]

        instance = MagicMock(autospec=TA2Core)
        instance.runs_dir = 'output/pipeline_runs'
        instance.folds = 5
        instance.store_summary = True
        instance.subprocess_evaluate.return_value = ([MagicMock()], all_results)

        pipeline = MagicMock()

        mock_file.reset_mock()
        TA2Core.score_pipeline(
            instance, {}, {'problem': {'performance_metrics': MagicMock()}}, pipeline
        )

        runs = [res.pipeline_run.to_json_structure() for res in all_results]

        mock_file.assert_called_once_with('output/pipeline_runs/{}.yml'.format(pipeline.id), 'w')
        mock_yaml.assert_called_once_with(runs, mock_file().__enter__(), default_flow_style=False)

    def test__check_stop_continue(self):
        instance = MagicMock(autospec=TA2Core)
        instance._stop = False
        instance.timeout = None
        TA2Core._check_stop(instance)

    def test__check_stop_true(self):
        instance = MagicMock(autospec=TA2Core)
        instance._stop = True
        with pytest.raises(KeyboardInterrupt):
            TA2Core._check_stop(instance)

    def test__check_stop_timeout(self):
        instance = MagicMock(autospec=TA2Core)
        instance._stop = False
        instance.timeout = 10
        instance.max_end_time = datetime.now() - timedelta(seconds=1)
        with pytest.raises(KeyboardInterrupt):
            TA2Core._check_stop(instance)

    def test_stop_subprocess(self):
        instance = MagicMock(autospec=TA2Core)
        instance.subprocess = Mock()

        TA2Core.stop(instance)

        assert instance._stop
        assert not instance.subprocess

    def test_stop_no_subprocess(self):
        instance = MagicMock(autospec=TA2Core)
        instance.subprocess = None

        TA2Core.stop(instance)

        assert instance._stop
        assert not instance.subprocess

    def test__setup_search(self):
        instance = MagicMock(autospec=TA2Core)

        TA2Core._setup_search(instance, None)

        assert instance.solutions == list()
        assert not instance._stop
        assert not instance.done
        assert not instance.timeout
        assert not instance.max_end_time

    def test__setup_search_with_timeout(self):
        instance = MagicMock(autospec=TA2Core)

        TA2Core._setup_search(instance, 5)

        assert instance.solutions == list()
        assert not instance._stop
        assert not instance.done
        assert instance.timeout == 5
        assert instance.max_end_time
