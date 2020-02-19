from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from ta3ta2_api import core_pb2_grpc

from ta2.ta3.core_servicer import CoreServicer, camel_case, dt2ts


def test_camel_case():
    assert camel_case('test') == 'test'
    assert camel_case('this is a test') == 'this is a test'
    assert camel_case('this_is_a_test') == 'thisIsATest'
    assert camel_case('THIS_IS_A_TEST') == 'THISISATEST'


def test_dt2ts():
    dt = datetime.strptime('21/11/06 16:30', '%d/%m/%y %H:%M')

    assert dt2ts(None) is None
    assert dt2ts(dt) == Timestamp(seconds=1164126600)


def test_core_servicer():
    input_dir = '/input'
    output_dir = '/output'
    static_dir = '/static'
    timeout = 0.5

    instance = CoreServicer(input_dir, output_dir, static_dir, timeout)
    assert isinstance(instance, core_pb2_grpc.CoreServicer)

    assert instance.input_dir == input_dir
    assert instance.output_dir == output_dir
    assert instance.static_dir == static_dir
    assert instance.timeout == timeout
    assert not instance.debug


@patch('ta2.ta3.core_servicer.LOGGER.exception')
@patch('ta2.ta3.core_servicer.LOGGER.info')
def test_core_servicer_run_session(logger_info_mock, logger_exception_mock):
    session = {'type': 'test-type', 'id': 'test-id'}

    # no arguments
    method = MagicMock()

    instance = CoreServicer('/input', '/output', '/static', 0.5)
    instance._run_session(session, method)

    args, kwargs = method.call_args

    assert method.call_count == 1
    assert logger_info_mock.call_count == 1
    assert logger_exception_mock.call_count == 0
    assert args == ()
    assert kwargs == {}
    assert 'end' in session
    assert 'done' in session
    assert 'error' not in session

    # with arguments
    instance._run_session(session, method, 'first-argument', second='argument')

    args, kwargs = method.call_args

    assert method.call_count == 2
    assert logger_info_mock.call_count == 2
    assert logger_exception_mock.call_count == 0
    assert args == ('first-argument',)
    assert kwargs == {'second': 'argument'}
    assert 'end' in session
    assert 'done' in session
    assert 'error' not in session

    # with exception
    method = MagicMock(side_effect=IndexError)
    instance._run_session(session, method)

    assert method.call_count == 1
    assert logger_info_mock.call_count == 3
    assert logger_exception_mock.call_count == 1
    assert 'end' in session
    assert 'done' in session
    assert 'error' in session


@patch('ta2.ta3.core_servicer.LOGGER.info')
@patch('ta2.ta3.core_servicer.threading.Thread')
def test_core_servicer_start_session(thread_mock, logger_mock):
    session_id = 'test-id'
    session_type = 'test-type'
    method = MagicMock()

    # debug mode
    instance = CoreServicer('/input', '/output', '/static', 0.5, debug=True)
    instance._start_session(session_id, session_type, method, 'first-argument', second='argument')

    args, kwargs = method.call_args

    assert len(instance.DB) == 1
    assert len(instance.DB[session_type + '_sessions']) == 1

    session = instance.DB[session_type + '_sessions'][session_id]

    assert method.call_count == 1
    assert logger_mock.call_count == 2
    assert args == ('first-argument',)
    assert kwargs == {}         # TODO: should kwargs be used?
    assert 'id' in session
    assert 'type' in session
    assert 'start' in session
    assert 'end' in session
    assert 'done' in session
    assert 'error' not in session

    # without debugging
    logger_mock.reset_mock()

    instance = CoreServicer('/input', '/output', '/static', 0.5)
    instance._start_session(session_id, session_type, method, 'first-argument', second='argument')

    assert len(instance.DB) == 1
    assert len(instance.DB[session_type + '_sessions']) == 1

    session = instance.DB[session_type + '_sessions'][session_id]
    expected_args = [session, method] + list(args)

    thread_mock.assert_called_once_with(target=instance._run_session, args=expected_args)
    assert logger_mock.call_count == 1
    assert 'id' in session
    assert 'type' in session
    assert 'start' in session

    # as thread is mocked `_run_session` is not called, therefore
    # `end` and `done` are not in session


@patch('ta2.ta3.core_servicer.Dataset.load')
@patch('ta2.ta3.core_servicer.decode_problem_description')
@patch('ta2.ta3.core_servicer.PipelineSearcher')
@patch('ta2.ta3.core_servicer.core_pb2.SearchSolutionsResponse')
def test_core_servicer_searchsolutions(searcher_mock, pipeline_searcher_mock, decode_mock, load_mock):
    instance = CoreServicer('/input', '/output', '/static', 0.5)
    instance._start_session = MagicMock()
    expected_result = 'result'
    searcher_mock.return_value = expected_result
    inputs = [MagicMock(dataset_uri=1)]

    # wrong version
    request = MagicMock(version='fake-version')

    with pytest.raises(AssertionError):
        instance.SearchSolutions(request, None)  # context (None) is not used

    # wrong problem inputs
    request = MagicMock(version='2020.1.28')

    with pytest.raises(AssertionError):
        instance.SearchSolutions(request, None)  # context (None) is not used

    # correct parameters
    problem = MagicMock(inputs=inputs)
    request = MagicMock(version='2020.1.28', inputs=inputs, problem=problem)

    result = instance.SearchSolutions(request, None)  # context (None) is not used

    decode_mock.assert_called_once_with(problem)
    pipeline_searcher_mock.assert_called_once_with(
        instance.input_dir, instance.output_dir, instance.static_dir)

    assert instance._start_session.call_count == 1
    assert result == expected_result


@patch('ta2.ta3.core_servicer.core_pb2.Progress')
@patch('ta2.ta3.core_servicer.core_pb2.ProgressState.Value')
def test_core_servicer_get_progress(progress_state_mock, progress_mock):
    instance = CoreServicer('/input', '/output', '/static', 0.5)

    # ERRORED
    session = {'error': 'test-value'}
    instance._get_progress(session)

    progress_state_mock.assert_called_with('ERRORED')
    assert progress_mock.call_count == 1

    # COMPLETED
    session = {'done': 'test-value'}
    instance._get_progress(session)

    progress_state_mock.assert_called_with('COMPLETED')
    assert progress_mock.call_count == 2

    # RUNNING
    session = {'other': 'test-value'}
    instance._get_progress(session)

    progress_state_mock.assert_called_with('RUNNING')
    assert progress_mock.call_count == 3


@patch('ta2.ta3.core_servicer.core_pb2.GetSearchSolutionsResultsResponse')
def test_core_servicer_get_search_soltuion_results(solutions_results_mock):
    instance = CoreServicer('/input', '/output', '/static', 0.5)
    instance._get_progress = MagicMock()
    solutions = {
        1: {'id': 1, 'score': 1, 'rank': 1},
        2: {'id': 2, 'score': 2, 'rank': 2}
    }

    # case 1: len(solutions) < returned
    searcher = MagicMock(solutions=solutions)
    session = {'searcher': searcher}

    result = instance._get_search_soltuion_results(session, 10)

    instance._get_progress.assert_not_called()
    solutions_results_mock.assert_not_called()

    assert result is None

    # case 2: len(solutions) > returned
    result = instance._get_search_soltuion_results(session, 1)

    instance._get_progress.assert_called_once()
    solutions_results_mock.assert_called_once()


def test_core_servicer_getsearchsolutionsresults():
    instance = CoreServicer('/input', '/output', '/static', 0.5)
    expected_result = 'result'
    instance._stream = MagicMock(return_value=expected_result)

    # invalid search_id
    request = MagicMock()

    with pytest.raises(ValueError):
        instance.GetSearchSolutionsResults(request, None)

    instance._stream.assert_not_called()

    # valid search_id
    search_id = 'test-id'
    instance.DB['search_sessions'] = {search_id: 'test-value'}
    request = MagicMock(search_id=search_id)

    result = instance.GetSearchSolutionsResults(request, None)

    instance._stream.assert_called_with('test-value', instance._get_search_soltuion_results)

    assert result == expected_result


@patch('ta2.ta3.core_servicer.core_pb2.EndSearchSolutionsResponse')
def test_core_servicer_endsearchsolutions(end_search_mock):
    instance = CoreServicer('/input', '/output', '/static', 0.5)
    expected_result = 'result'
    end_search_mock.return_value = expected_result

    search_id = 'test-id'
    request = MagicMock(search_id=search_id)

    # empty session
    instance.DB['search_sessions'] = {search_id: {}}

    result = instance.EndSearchSolutions(request, None)
    end_search_mock.assert_called_once()

    assert result == expected_result

    # session with searcher
    searcher = MagicMock(done=True, solutions={})
    searcher.stop = MagicMock()

    instance.DB['search_sessions'] = {search_id: {'searcher': searcher}}

    result = instance.EndSearchSolutions(request, None)

    searcher.stop.assert_called_once()
    assert result == expected_result
    assert end_search_mock.call_count == 2


@patch('ta2.ta3.core_servicer.core_pb2.StopSearchSolutionsResponse')
def test_core_servicer_stopsearchsolutions(stop_search_mock):
    instance = CoreServicer('/input', '/output', '/static', 0.5)
    expected_result = 'result'
    stop_search_mock.return_value = expected_result

    search_id = 'test-id'
    request = MagicMock(search_id=search_id)

    # empty session
    instance.DB['search_sessions'] = {search_id: {}}

    result = instance.StopSearchSolutions(request, None)

    stop_search_mock.assert_called_once()

    assert result == expected_result

    # session with searcher
    searcher = MagicMock(done=True)
    searcher.stop = MagicMock()

    instance.DB['search_sessions'] = {search_id: {'searcher': searcher}}

    instance.StopSearchSolutions(request, None)

    searcher.stop.assert_called_once()
    assert result == expected_result
    assert stop_search_mock.call_count == 2
