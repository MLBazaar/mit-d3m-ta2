from datetime import datetime
from unittest.mock import MagicMock, patch

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
    input_dir = '/input-dir'
    output_dir = '/output-dir'
    timeout = 0.5

    instance = CoreServicer(input_dir, output_dir, timeout)
    assert isinstance(instance, core_pb2_grpc.CoreServicer)

    assert instance.input_dir == input_dir
    assert instance.output_dir == output_dir
    assert instance.timeout == timeout
    assert not instance.debug


@patch('ta2.ta3.core_servicer.LOGGER.exception')
@patch('ta2.ta3.core_servicer.LOGGER.info')
def test_core_servicer_run_session(logger_info_mock, logger_exception_mock):
    session = {'type': 'test-type', 'id': 'test-id'}

    # no arguments
    method = MagicMock()

    instance = CoreServicer('/input-dir', '/output-dir', 0.5)
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
