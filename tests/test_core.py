from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import numpy as np

from ta2.core import (DATAMART_URL, DATA_PIPELINE_PATH, DEFAULT_SCORING_PIPELINE_PATH,
                      TEMPLATES_DIR, ScoringError, TA2Core)


class TestTA2Core(TestCase):

    @patch('ta2.core.os')
    @patch('ta2.core.json')
    @patch('ta2.core.open')
    def test__valid_template_invalid(self, mock_open, mock_json, mock_os):
        # setup
        instance = MagicMock(autospec=TA2Core)
        mock_json.load.side_effect = [Exception()]
        mock_os.path.join.return_value = 'abs_path/demo-template.json'

        # run
        result = TA2Core._valid_template(instance, 'demo-template.json')

        # assert
        assert not result
        mock_os.path.join.assert_called_once_with(TEMPLATES_DIR, 'demo-template.json')
        mock_open.assert_called_once_with('abs_path/demo-template.json', 'r')
        mock_json.load.assert_called_once_with(mock_open.return_value.__enter__())

    @patch('ta2.core.os')
    @patch('ta2.core.json')
    @patch('ta2.core.open')
    def test__valid_template_valid(self, mock_open, mock_json, mock_os):
        # setup
        instance = MagicMock(autospec=TA2Core)
        mock_os.path.join.return_value = 'abs_path/demo-template.json'

        # run
        result = TA2Core._valid_template(instance, 'demo-template.json')

        # assert
        assert result
        mock_os.path.join.assert_called_once_with(TEMPLATES_DIR, 'demo-template.json')
        mock_open.assert_called_once_with('abs_path/demo-template.json', 'r')
        mock_json.load.assert_called_once_with(mock_open.return_value.__enter__())

    def test__select_templates(self):
        pass  # TODO

    @patch('random.sample')
    @patch('ta2.core.filter')
    @patch('ta2.core.os')
    def test__get_all_templates(self, mock_os, mock_filter, mock_sample):
        # setup
        instance = MagicMock(autospec=TA2Core)

        mock_os.listdir.return_value = ['template-1.json', 'template-2.json', 'template-3.json']
        mock_filter.return_value = ['template-1.json', 'template-2.json', 'template-3.json']
        mock_sample.return_value = ['template-2.json', 'template-3.json', 'template-1.json']

        # run
        result = TA2Core._get_all_templates(instance)

        # assert
        expected_sample_list = ['template-1.json', 'template-2.json', 'template-3.json']

        mock_filter.assert_called_once_with(instance._valid_template, mock_os.listdir.return_value)
        mock_sample.assert_called_once_with(expected_sample_list, len(expected_sample_list))

        assert set(result) == set(['template-1.json', 'template-2.json', 'template-3.json'])

    @patch('ta2.core.load_pipeline')
    @patch('ta2.core.os')
    def test___init__defaults(self, mock_os, mock_load_pipeline):
        # setup
        mock_os.path.join.side_effect = [
            'output/pipeline_runs',
            'output/pipelines_ranked',
            'output/pipelines_scored',
            'output/pipelines_searched'
        ]

        mock_load_pipeline.side_effect = ['data_pipeline', 'default_scoring_pipeline']

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

        assert mock_os.makedirs.call_args_list == expected_makedirs_calls

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

        assert mock_load_pipeline.call_args_list == expceted_load_pipeline_calls

    @patch('ta2.core.load_pipeline')
    @patch('ta2.core.os')
    def test___init__not_defaults(self, mock_os, mock_load_pipeline):
        # setup
        mock_os.path.join.side_effect = [
            'new-output/pipeline_runs',
            'new-output/pipelines_ranked',
            'new-output/pipelines_scored',
            'new-output/pipelines_searched'
        ]

        mock_load_pipeline.side_effect = ['data_pipeline', 'default_scoring_pipeline']

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

        assert mock_os.makedirs.call_args_list == expected_makedirs_calls

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

        assert mock_load_pipeline.call_args_list == expceted_load_pipeline_calls

    @patch('ta2.core.d3m_evaluate')
    def test__evaluate(self, mock_d3m_evaluate):
        # setup
        out = Mock()
        pipeline = Mock()

        # run
        TA2Core._evaluate(out, pipeline)

        # assert
        mock_d3m_evaluate.assert_called_once_with(pipeline)
        out.extend.assert_called_once_with(mock_d3m_evaluate.return_value)

    def test_subprocess_evaluate(self):
        pass  # TODO

    def test_score_pipeline_failed(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.folds = 5
        instance.subprocess_evaluate.return_value = (False, [MagicMock()])

        # run
        dataset = dict()
        problem = {'problem': {'performance_metrics': None}}
        pipeline = MagicMock()

        with pytest.raises(ScoringError):
            TA2Core.score_pipeline(instance, dataset, problem, pipeline)

    @patch('yaml.dump_all')
    @patch('ta2.core.open')
    def test_score_pipeline_dump_summary(self, mock_file, mock_yaml):
        # setup
        all_results = [Mock(), Mock()]

        score_1 = MagicMock()
        score_1.value = [1, 1, 1]

        score_2 = MagicMock()
        score_2.value = [2, 2, 2]

        all_scores = [score_1, score_2]

        instance = MagicMock(autospec=TA2Core)
        instance.runs_dir = 'output/pipeline_runs'
        instance.folds = 5
        instance.store_summary = True
        instance.subprocess_evaluate.return_value = (all_scores, all_results)

        # run
        dataset = dict()
        metric = Mock()
        problem = {'problem': {'performance_metrics': [{'metric': metric}]}}
        pipeline = MagicMock()

        TA2Core.score_pipeline(
            instance, dataset, problem, pipeline
        )

        # assert
        expected_runs = [
            all_results[0].pipeline_run.to_json_structure(),
            all_results[1].pipeline_run.to_json_structure()
        ]

        assert pipeline.cv_scores == [1, 2]
        assert pipeline.score == 1.5

        mock_file.assert_called_once_with('output/pipeline_runs/{}.yml'.format(pipeline.id), 'w')
        mock_yaml.assert_called_once_with(
            expected_runs,
            mock_file().__enter__(),
            default_flow_style=False
        )

        metric.normalize.assert_called_once_with(pipeline.score)

    @patch('ta2.core.dump_pipeline')
    def test__save_pipeline_none_score(self, mock_dump_pipeline):
        # setup
        instance = MagicMock(autospec=TA2Core)

        pipeline = Mock()
        pipeline.score = None
        pipeline.to_json_structure.return_value = dict()

        # run
        TA2Core._save_pipeline(instance, pipeline)

        # assert
        pipeline.to_json_structure.assert_called_once_with()
        mock_dump_pipeline.assert_called_once_with(dict(), instance.searched_dir)

    @patch('ta2.core.random')
    @patch('ta2.core.dump_pipeline')
    def test__save_pipeline_with_score_no_dump(self, mock_dump_pipeline, mock_random):

        def save_original_call(pipeline_dict, scored_dir):
            self.called_with_pp_dict = pipeline_dict.copy()
            self.called_with_scored_dir = scored_dir

        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.dump = False
        mock_dump_pipeline.side_effect = save_original_call

        pipeline = MagicMock()
        pipeline.score = 0.5
        pipeline.normalized_score = 0.5
        pipeline.to_json_structure.return_value = dict()

        mock_random.random.return_value = 0.1

        # run
        TA2Core._save_pipeline(instance, pipeline)

        # assert
        expected_pipeline_dict = {
            'rank': (1 - 0.5) + 0.1 * 1.e-12,
            'score': 0.5,
            'normalized_score': 0.5
        }

        assert self.called_with_pp_dict == dict()
        pipeline.to_json_structure.assert_called_once_with()
        instance.solutions.append.assert_called_once_with(expected_pipeline_dict)

    @patch('ta2.core.random')
    @patch('ta2.core.dump_pipeline')
    def test__save_pipeline_with_score_dump(self, mock_dump_pipeline, mock_random):
        self.called_with_pp_dict = list()
        self.called_with_dirs = list()
        self.ranks = list()

        def save_original_call(pipeline_dict, scored_dir, rank=None):
            self.called_with_pp_dict.append(pipeline_dict.copy())
            self.called_with_dirs.append(scored_dir)
            if rank:
                self.ranks.append(rank)

        # setup
        mock_dump_pipeline.side_effect = save_original_call

        instance = MagicMock(autospec=TA2Core)
        instance.dump = True

        pipeline = Mock()
        pipeline.score = 0.5
        pipeline.normalized_score = 0.5
        pipeline.to_json_structure.return_value = dict()

        mock_random.random.return_value = 0.1

        # run
        TA2Core._save_pipeline(instance, pipeline)

        # assert
        expected_pipeline_dict = {
            'rank': (1 - 0.5) + 0.1 * 1.e-12,
            'score': 0.5,
            'normalized_score': 0.5
        }

        expected_dump_pipeline_dicts = [dict(), dict()]
        expected_dump_pipeline_dirs = [instance.scored_dir, instance.ranked_dir]
        expected_dump_pipeline_ranks = [(1 - 0.5) + 0.1 * 1.e-12]

        assert self.called_with_pp_dict == expected_dump_pipeline_dicts
        assert self.called_with_dirs == expected_dump_pipeline_dirs
        assert self.ranks == expected_dump_pipeline_ranks

        pipeline.to_json_structure.assert_called_once_with()
        instance.solutions.append.assert_called_once_with(expected_pipeline_dict)

    def test__new_pipeline(self):
        pass  # TODO

    def test__check_stop_continue(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance._stop = False
        instance.timeout = None

        # run
        TA2Core._check_stop(instance)

    def test__check_stop_true(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance._stop = True

        # run
        with pytest.raises(KeyboardInterrupt):
            TA2Core._check_stop(instance)

    def test__check_stop_timeout(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance._stop = False
        instance.timeout = 10
        instance.max_end_time = datetime.now() - timedelta(seconds=1)

        # run
        with pytest.raises(KeyboardInterrupt):
            TA2Core._check_stop(instance)

    def test_stop_subprocess(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.subprocess = Mock()

        # run
        TA2Core.stop(instance)

        # assert
        assert instance._stop
        assert instance.subprocess is None

    def test_stop_no_subprocess(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.subprocess = None

        # run
        TA2Core.stop(instance)

        # assert
        assert instance._stop
        assert instance.subprocess is None

    def test__timeout(self):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.killed = False

        # run
        with pytest.raises(KeyboardInterrupt):
            TA2Core._timeout(instance)
            assert instance.killed

    @patch('ta2.core.datetime')
    def test__setup_search_without_timeout(self, mock_datetime):
        # setup
        instance = MagicMock(autospec=TA2Core)

        datetime_now = datetime.now()
        mock_datetime.now.return_value = datetime_now

        # run
        TA2Core._setup_search(instance, None)

        # assert
        assert instance.timeout is None
        assert instance.solutions == list()
        assert instance.summary == list()
        assert not instance._stop
        assert not instance.done
        assert instance.start_time == datetime_now
        assert instance.max_end_time is None
        assert not instance.killed
        assert instance.best_pipeline is None
        assert instance.best_score is None
        assert instance.best_normalized == -np.inf
        assert instance.best_template_name is None
        assert instance.spent == 0
        assert instance.iterations == 0
        assert instance.scored == 0
        assert instance.errored == 0
        assert instance.invalid == 0
        assert instance.timedout == 0

    @patch('ta2.core.signal')
    @patch('ta2.core.datetime')
    def test__setup_search_with_timeout(self, mock_datetime, mock_signal):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.hard_timeout = False

        datetime_now = datetime.now()
        mock_datetime.now.return_value = datetime_now

        # run
        TA2Core._setup_search(instance, 5)

        # assert
        expected_max_end_time = datetime_now + timedelta(seconds=5)

        assert instance.timeout is 5
        assert instance.solutions == list()
        assert instance.summary == list()
        assert not instance._stop
        assert not instance.done
        assert instance.start_time == datetime_now
        assert instance.max_end_time == expected_max_end_time
        assert not instance.killed
        assert instance.best_pipeline is None
        assert instance.best_score is None
        assert instance.best_normalized == -np.inf
        assert instance.best_template_name is None
        assert instance.spent == 0
        assert instance.iterations == 0
        assert instance.scored == 0
        assert instance.errored == 0
        assert instance.invalid == 0
        assert instance.timedout == 0

        mock_signal.signal.assert_not_called()
        mock_signal.alarm.assert_not_called()

    @patch('ta2.core.signal')
    @patch('ta2.core.datetime')
    def test__setup_search_with_timeout_hard_timeout(self, mock_datetime, mock_signal):
        # setup
        instance = MagicMock(autospec=TA2Core)
        instance.hard_timeout = True

        datetime_now = datetime.now()
        mock_datetime.now.return_value = datetime_now

        # run
        TA2Core._setup_search(instance, 5)

        # assert
        expected_max_end_time = datetime_now + timedelta(seconds=5)

        assert instance.timeout is 5
        assert instance.solutions == list()
        assert instance.summary == list()
        assert not instance._stop
        assert not instance.done
        assert instance.start_time == datetime_now
        assert instance.max_end_time == expected_max_end_time
        assert not instance.killed
        assert instance.best_pipeline is None
        assert instance.best_score is None
        assert instance.best_normalized == -np.inf
        assert instance.best_template_name is None
        assert instance.spent == 0
        assert instance.iterations == 0
        assert instance.scored == 0
        assert instance.errored == 0
        assert instance.invalid == 0
        assert instance.timedout == 0

        mock_signal.signal.assert_called_once_with(mock_signal.SIGALRM, instance._timeout)
        mock_signal.alarm.assert_called_once_with(5)

    @patch('ta2.core.DatamartQuery')
    @patch('ta2.core.RESTDatamart')
    def test__get_data_augmentation_no_aumentation(self, mock_rest_datamart, mock_datamart_query):
        # setup
        instance = MagicMock(autospec=TA2Core)

        datamart = Mock()

        cursor = Mock()
        datamart.return_value = cursor

        mock_rest_datamart.return_value = datamart

        # run
        result = TA2Core._get_data_augmentation(instance, dict(), dict())

        # assert
        assert result is None

        mock_rest_datamart.assert_called_once_with(DATAMART_URL)
        mock_datamart_query.assert_not_called()
        datamart.search_with_data.assert_not_called()
        cursor.get_next_page.assert_not_called()

    @patch('ta2.core.DatamartQuery')
    @patch('ta2.core.RESTDatamart')
    def test__get_data_augmentation_with_next_page(self, mock_rest_datamart, mock_datamart_query):
        # setup
        instance = MagicMock(autospec=TA2Core)

        page_0 = Mock()
        page_0.serialize.return_value = 'serialize'
        page = [page_0]

        cursor = Mock()
        cursor.get_next_page.return_value = page

        datamart = Mock()
        datamart.search_with_data.return_value = cursor

        mock_rest_datamart.return_value = datamart

        query = Mock()
        mock_datamart_query.return_value = query

        # run
        problem = {
            'data_augmentation': [
                {
                    'keywords': ['foo', 'bar']
                }
            ]
        }
        result = TA2Core._get_data_augmentation(instance, dict(), problem)

        # assert
        assert result == 'serialize'

        mock_rest_datamart.assert_called_once_with(DATAMART_URL)
        mock_datamart_query.assert_called_once_with(keywords=['foo', 'bar'])
        datamart.search_with_data.assert_called_once_with(query=query, supplied_data=dict())
        cursor.get_next_page.assert_called_once_with()

    @patch('ta2.core.DatamartQuery')
    @patch('ta2.core.RESTDatamart')
    def test__get_data_augmentation_without_next_page(
            self, mock_rest_datamart, mock_datamart_query):

        # setup
        instance = MagicMock(TA2Core)

        cursor = Mock()
        cursor.get_next_page.return_value = list()

        datamart = Mock()
        datamart.search_with_data.return_value = cursor

        mock_rest_datamart.return_value = datamart

        query = Mock()
        mock_datamart_query.return_value = query

        # run
        problem = {
            'data_augmentation': [
                {
                    'keywords': ['foo', 'bar']
                }
            ]
        }
        result = TA2Core._get_data_augmentation(instance, dict(), problem)

        # assert
        assert result is None

        mock_rest_datamart.assert_called_once_with(DATAMART_URL)
        mock_datamart_query.assert_called_once_with(keywords=['foo', 'bar'])
        datamart.search_with_data.assert_called_once_with(query=query, supplied_data=dict())
        cursor.get_next_page.assert_called_once_with()

    def test__make_btb_scorer(self):
        pass  # TODO
