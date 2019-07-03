from unittest.mock import MagicMock, patch

from ta2.ta3.server import serve


@patch('ta2.ta3.client.LOGGER.info')
@patch('ta2.ta3.server.time.sleep')
@patch('ta2.ta3.server.core_pb2_grpc.add_CoreServicer_to_server')
@patch('ta2.ta3.server.grpc.server')
@patch('ta2.ta3.server.core_servicer.CoreServicer')
def test_serve(core_servicer_mock, grpc_server_mock, add_cs_to_server_mock, sleep_mock, logger_mock):
    # mocks
    expected_value = MagicMock()
    grpc_server_mock.return_value = expected_value

    # arguments
    port = 9999
    input_dir = 'input'
    output_dir = 'output'
    static_dir = 'static'
    timeout = 60
    debug = False

    # daemon=True
    return_value = serve(port, input_dir, output_dir, static_dir, timeout, debug, daemon=True)
    core_servicer_mock.assert_called_once_with(input_dir, output_dir, static_dir, timeout, debug)

    assert return_value == expected_value
    assert grpc_server_mock.called
    assert not sleep_mock.called
    assert not logger_mock.called
