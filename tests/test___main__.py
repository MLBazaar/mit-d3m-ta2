from unittest.mock import MagicMock, call, patch

import pytest

from ta2 import __main__


@patch('ta2.__main__._server')
@patch('ta2.__main__.parse_args')
def test_ta2_server(mock_parse_args, mock__server):
    # run
    __main__.ta2_server()

    # assert
    mock_parse_args.assert_called_once_with('server')
    mock__server.assert_called_once_with(mock_parse_args.return_value)


@patch('ta2.__main__._ta2_test')
@patch('ta2.__main__.parse_args')
def test_ta2_test(mock_parse_args, mock__ta2_test):
    # run
    __main__.ta2_test()

    # assert
    mock_parse_args.assert_called_once_with('ta2')
    mock__ta2_test.assert_called_once_with(mock_parse_args.return_value)
