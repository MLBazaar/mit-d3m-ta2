from unittest.mock import MagicMock, mock_open, patch

from ta2_test import load_dataset, load_pipeline, load_problem, search


@patch('ta2_test.Dataset.load')
def test_load_dataset(loader_mock, tmp_path):
    expected_result = 'result'
    loader_mock.return_value = expected_result

    root_path = tmp_path / 'root_path'
    phase = 'phase'
    inner_phase = 'inner_phase'

    # without inner_phase (None)
    uri = 'file://{}/{}/dataset_{}/datasetDoc.json'.format(root_path, phase, phase)
    result = load_dataset(root_path, phase)

    assert result == expected_result
    loader_mock.assert_called_once_with(dataset_uri=uri)

    # with inner_phase
    uri = 'file://{}/{}/dataset_{}/datasetDoc.json'.format(root_path, phase, inner_phase)
    result = load_dataset(root_path, phase, inner_phase)

    assert result == expected_result
    loader_mock.assert_called_with(dataset_uri=uri)


@patch('ta2_test.Problem.load')
def test_load_problem(loader_mock, tmp_path):
    expected_result = 'result'
    loader_mock.return_value = expected_result

    root_path = tmp_path / 'root_path'
    phase = 'phase'

    uri = 'file://{}/{}/problem_{}/problemDoc.json'.format(root_path, phase, phase)
    result = load_problem(root_path, phase)

    assert result == expected_result
    loader_mock.assert_called_once_with(problem_uri=uri)


@patch('ta2_test.Pipeline.from_yaml')
@patch('ta2_test.Pipeline.from_json')
def test_load_pipeline(json_loader_mock, yaml_loader_mock):
    json_loader_mock.return_value = 'json-file'
    yaml_loader_mock.return_value = 'yaml-file'
    open_mock = mock_open(read_data='data')

    # json file
    with patch('ta2_test.open', open_mock) as mock:
        result = load_pipeline('test.json')

    assert result == 'json-file'
    mock.assert_called_once_with('test.json', 'r')
    assert json_loader_mock.call_count == 1

    # other file
    with patch('ta2_test.open', open_mock) as mock:
        result = load_pipeline('test.ext')

    assert result == 'yaml-file'
    mock.assert_called_with('test.ext', 'r')
    assert yaml_loader_mock.call_count == 1


@patch('ta2_test.PipelineSearcher')
def test_search(searcher_mock):
    instance_mock = MagicMock()
    searcher_mock.return_value = instance_mock
    expected_result = 'result'
    instance_mock.search = MagicMock(return_value=expected_result)

    dataset_root = 'dataset'
    problem = MagicMock()
    args = MagicMock(input='input', output='output', timeout='timeout', budget='budget')

    result = search(dataset_root, problem, args)

    assert result == expected_result
    searcher_mock.assert_called_once_with('input', 'output', dump=True)
    instance_mock.search.assert_called_once_with(problem, timeout='timeout', budget='budget')
