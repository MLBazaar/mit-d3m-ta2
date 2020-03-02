from unittest import TestCase
from unittest.mock import Mock, patch

from ta2 import __main__


class TestMain(TestCase):

    @patch('ta2.__main__._jupyter_docker')
    def test__jupyter_environment_docker(self, mock_jupyter_docker):
        """Run jupyter notebook on Docker mode."""
        args = Mock()
        args.environment = 'docker'

        __main__._jupyter(args)
        mock_jupyter_docker.assert_called_once_with(args)

    @patch('ta2.__main__._jupyter_native')
    def test__jupyter_environment_native(self, mock_jupyter_native):
        """Run jupyter notebook on Native mode."""
        args = Mock()
        args.environment = 'native'

        __main__._jupyter(args)
        mock_jupyter_native.assert_called_once_with(args)

    @patch('ta2.__main__._run_docker')
    def test__ta2_standalone_docker(self, mock_standalone_docker):
        """Run ta2 standalone on Docker mode."""
        args = Mock()
        args.environment = 'docker'

        __main__._ta2_standalone(args)
        mock_standalone_docker.assert_called_once_with(args)

    @patch('ta2.__main__._standalone_native')
    def test__ta2_standalone_native(self, mock_standalone_native):
        """Run ta2 standalone on Native mode."""
        args = Mock()
        args.environment = 'native'

        __main__._ta2_standalone(args)
        mock_standalone_native.assert_called_once_with(args)
