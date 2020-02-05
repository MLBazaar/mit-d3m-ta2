from os.path import abspath
from unittest.mock import MagicMock, patch

from ta3ta2_api.value_pb2 import Value, ValueType

from ta2.ta3.client import TA3APIClient, pythonize


def test_pythonize():
    assert pythonize('test') == 'TEST'
    assert pythonize('this is a test') == 'THIS IS A TEST'
    assert pythonize('this_is_a_test_0') == 'THIS_IS_A_TEST_0'


def test_apiclient_get_dataset_doc_path():
    # with default args
    remote_input = 'input'
    instance = TA3APIClient(port=9999)
    dataset = 'test_dataset'

    doc_path = instance._get_dataset_doc_path(dataset)
    expected_doc_path = 'file://{}/{}/{}/TRAIN/dataset_TRAIN/datasetDoc.json'.format(
        abspath('.'), remote_input, dataset
    )

    assert doc_path == expected_doc_path

    # custom args
    remote_input = 'my-remote-input'
    instance = TA3APIClient(port=9999, remote_input=remote_input)

    doc_path = instance._get_dataset_doc_path(dataset)
    expected_doc_path = 'file://{}/{}/{}/TRAIN/dataset_TRAIN/datasetDoc.json'.format(
        abspath('.'), remote_input, dataset
    )

    assert doc_path == expected_doc_path


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.SearchSolutionsRequest')
def test_apiclient_search_solutions(search_solutions_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    dataset = 'test_dataset'
    expected_value = 'response'

    # instance mocks
    instance._get_dataset_doc_path = MagicMock(return_value='dataset-doc-path')
    instance._build_problem = MagicMock(return_value='build-problem')
    instance.stub.SearchSolutions = MagicMock(return_value=expected_value)

    value = instance.search_solutions(dataset)
    assert value == expected_value

    search_solutions_request_mock.assert_called_once_with(
        user_agent='ta3_api_test.py',
        version='2020.1.28',
        time_bound_search=1.,
        priority=0.,
        allowed_value_types=[
            ValueType.Value('RAW'),
            ValueType.Value('DATASET_URI'),
            ValueType.Value('CSV_URI'),
        ],
        inputs=[
            Value(dataset_uri='dataset-doc-path')
        ],
        problem='build-problem'
    )

    assert logger_mock.call_count == 2


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.GetSearchSolutionsResultsRequest')
def test_apiclient_get_search_solutions_results(get_search_solutions_results_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    search_id = 'search-id'

    # 1. no solutions
    expected_value = []

    # instance mocks
    instance.stub.GetSearchSolutionsResults = MagicMock(return_value=expected_value)

    value = instance.get_search_solutions_results(search_id)

    get_search_solutions_results_request_mock.assert_called_once_with(search_id=search_id)

    assert value == expected_value
    assert logger_mock.call_count == len(expected_value) + 1

    # 2. two solutions
    logger_mock.reset_mock()

    expected_value = [1, 2]

    # instance mocks
    instance.stub.GetSearchSolutionsResults = MagicMock(return_value=expected_value)

    value = instance.get_search_solutions_results(search_id)

    get_search_solutions_results_request_mock.assert_called_with(search_id=search_id)

    assert value == expected_value
    assert logger_mock.call_count == len(expected_value) + 1

    # 3. two solutions but max one result
    logger_mock.reset_mock()

    expected_value = [1]

    # instance mocks
    instance.stub.GetSearchSolutionsResults = MagicMock(return_value=expected_value + [2, 3])

    value = instance.get_search_solutions_results(search_id, max_results=1)

    get_search_solutions_results_request_mock.assert_called_with(search_id=search_id)

    assert value == expected_value
    assert logger_mock.call_count == len(expected_value) + 1


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.EndSearchSolutionsRequest')
def test_apiclient_end_search_solutions(end_search_solutions_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    search_id = 'search-id'
    expected_response = 'response'

    # instance mocks
    instance.stub.EndSearchSolutions = MagicMock(return_value=expected_response)

    end_search_solutions_request_mock.return_value = 'request'

    return_value = instance.end_search_solutions(search_id)

    assert return_value == expected_response
    end_search_solutions_request_mock.assert_called_once_with(search_id=search_id)
    instance.stub.EndSearchSolutions.called_once_with('request')
    assert logger_mock.call_count == 2


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.StopSearchSolutionsRequest')
def test_apiclient_stop_search_solutions(stop_search_solutions_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    search_id = 'search-id'
    expected_response = 'response'

    # instance mocks
    instance.stub.StopSearchSolutions = MagicMock(return_value=expected_response)

    stop_search_solutions_request_mock.return_value = 'request'

    return_value = instance.stop_search_solutions(search_id)

    assert return_value == expected_response
    stop_search_solutions_request_mock.assert_called_once_with(search_id=search_id)
    instance.stub.StopSearchSolutions.called_once_with('request')
    assert logger_mock.call_count == 2


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.DescribeSolutionRequest')
def test_apiclient_describe_solution(describe_solution_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    solution_id = 'solution-id'
    expected_response = 'response'

    # instance mocks
    instance.stub.DescribeSolution = MagicMock(return_value=expected_response)

    describe_solution_request_mock.return_value = 'request'

    return_value = instance.describe_solution(solution_id)

    assert return_value == expected_response
    describe_solution_request_mock.assert_called_once_with(solution_id=solution_id)
    instance.stub.DescribeSolution.called_once_with('request')
    assert logger_mock.call_count == 2


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.ScoreSolutionRequest')
def test_apiclient_score_solution(score_solution_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    solution_id = 'solution-id'
    dataset = 'test-dataset'
    expected_response = 'response'

    # mocks
    score_solution_request_mock.return_value = 'request'
    problem_mock = MagicMock()
    problem_mock.problem.performance_metrics = 'metrics'
    instance._build_problem = MagicMock(return_value=problem_mock)
    instance._get_dataset_doc_path = MagicMock(return_value='dataset-doc-path')
    instance.stub.ScoreSolution = MagicMock(return_value=expected_response)

    return_value = instance.score_solution(solution_id, dataset)

    assert return_value == expected_response
    assert logger_mock.call_count == 2
    assert score_solution_request_mock.called

    instance.stub.ScoreSolution.called_once_with('request')


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.GetScoreSolutionResultsRequest')
def test_apiclient_get_score_solution(get_score_solution_results_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    request_id = 'request-id'

    # 1. no scores
    scores = []

    # mocks
    get_score_solution_results_request_mock.return_value = 'request'
    instance.stub.GetScoreSolutionResults = MagicMock(return_value=scores)

    value = instance.get_score_solution_results(request_id)

    get_score_solution_results_request_mock.assert_called_once_with(request_id=request_id)
    instance.stub.GetScoreSolutionResults.assert_called_once_with('request')

    assert value is None
    assert logger_mock.call_count == len(scores) + 1

    # 2. three solutions
    logger_mock.reset_mock()
    scores = [1, 2, 3]

    # mocks
    instance.stub.GetScoreSolutionResults = MagicMock(return_value=scores)

    value = instance.get_score_solution_results(request_id)

    get_score_solution_results_request_mock.assert_called_with(request_id=request_id)
    instance.stub.GetScoreSolutionResults.assert_called_once_with('request')

    assert value == scores[-1]           # the last score
    assert logger_mock.call_count == len(scores) + 1


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.FitSolutionRequest')
def test_apiclient_fit_solution(fit_solution_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    solution_id = 'solution-id'
    dataset = 'test-dataset'
    expected_response = 'response'

    # mocks
    fit_solution_request_mock.return_value = 'request'
    instance._get_dataset_doc_path = MagicMock(return_value='dataset-doc-path')
    instance.stub.FitSolution = MagicMock(return_value=expected_response)

    return_value = instance.fit_solution(solution_id, dataset)

    assert return_value == expected_response
    assert logger_mock.call_count == 2

    fit_solution_request_mock.assert_called_once_with(
        solution_id=solution_id,
        inputs=[
            Value(dataset_uri='dataset-doc-path')
        ],
        expose_outputs=[
            'outputs.0'
        ],
        expose_value_types=[
            ValueType.Value('CSV_URI')
        ]
    )

    instance.stub.FitSolution.called_once_with('request')


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.GetFitSolutionResultsRequest')
def test_apiclient_get_fit_solution_results(get_fit_solution_results_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    request_id = 'request-id'

    # 1. no solutions
    expected_value = []

    # mocks
    get_fit_solution_results_request_mock.return_value = 'request'
    instance.stub.GetFitSolutionResults = MagicMock(return_value=expected_value)

    value = instance.get_fit_solution_results(request_id)

    get_fit_solution_results_request_mock.assert_called_once_with(request_id=request_id)
    instance.stub.GetFitSolutionResults.assert_called_once_with('request')

    assert value == expected_value
    assert logger_mock.call_count == len(expected_value) + 1

    # 2. two solutions
    logger_mock.reset_mock()

    expected_value = [1, 2]

    # mocks
    instance.stub.GetFitSolutionResults = MagicMock(return_value=expected_value)

    value = instance.get_fit_solution_results(request_id)

    get_fit_solution_results_request_mock.assert_called_with(request_id=request_id)
    instance.stub.GetFitSolutionResults.assert_called_once_with('request')

    assert value == expected_value
    assert logger_mock.call_count == len(expected_value) + 1

    # 3. two solutions but max one result
    logger_mock.reset_mock()

    expected_value = [1]

    # mocks
    instance.stub.GetFitSolutionResults = MagicMock(return_value=expected_value + [2, 3])

    value = instance.get_fit_solution_results(request_id, max_results=1)

    get_fit_solution_results_request_mock.assert_called_with(request_id=request_id)
    instance.stub.GetFitSolutionResults.assert_called_once_with('request')

    assert value == expected_value
    assert logger_mock.call_count == len(expected_value) + 1


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.ProduceSolutionRequest')
def test_apiclient_produce_solution(produce_solution_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    solution_id = 'solution-id'
    dataset = 'test-dataset'
    expected_response = 'response'

    # mocks
    produce_solution_request_mock.return_value = 'request'
    instance._get_dataset_doc_path = MagicMock(return_value='dataset-doc-path')
    instance.stub.ProduceSolution = MagicMock(return_value=expected_response)

    return_value = instance.produce_solution(solution_id, dataset)

    assert return_value == expected_response
    assert logger_mock.call_count == 2

    produce_solution_request_mock.assert_called_once_with(
        fitted_solution_id=solution_id,
        inputs=[
            Value(dataset_uri='dataset-doc-path')
        ],
        expose_outputs=[
            'outputs.0'
        ],
        expose_value_types=[
            ValueType.Value('CSV_URI')
        ]
    )

    instance.stub.ProduceSolution.called_once_with('request')


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.GetProduceSolutionResultsRequest')
def test_apiclient_get_produce_solution_results(get_produce_solution_results_request_mock, logger_mock):
    # TODO: this method might have a bug - results are not being added to the `results` variable
    pass


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.SolutionExportRequest')
def test_apiclient_solution_export(solution_export_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    solution_id = 'solution-id'
    rank = 'rank'
    expected_response = 'response'

    # mocks
    solution_export_request_mock.return_value = 'request'
    instance.stub.SolutionExport = MagicMock(return_value=expected_response)

    return_value = instance.solution_export(solution_id, rank)

    assert return_value == expected_response
    assert logger_mock.call_count == 2

    solution_export_request_mock.assert_called_once_with(solution_id=solution_id, rank=rank)
    instance.stub.SolutionExport.called_once_with('request')


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.ListPrimitivesRequest')
def test_apiclient_list_primitives(list_primitives_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    expected_response = 'response'

    # mocks
    list_primitives_request_mock.return_value = 'request'
    instance.stub.ListPrimitives = MagicMock(return_value=expected_response)

    return_value = instance.list_primitives()

    assert return_value == expected_response
    assert logger_mock.call_count == 2

    list_primitives_request_mock.assert_called_once_with()
    instance.stub.ListPrimitives.called_once_with('request')


@patch('ta2.ta3.client.LOGGER.debug')
@patch('ta2.ta3.client.core_pb2.HelloRequest')
def test_apiclient_hello(hello_request_mock, logger_mock):
    instance = TA3APIClient(port=9999)
    expected_response = 'response'

    # mocks
    hello_request_mock.return_value = 'request'
    instance.stub.Hello = MagicMock(return_value=expected_response)

    return_value = instance.hello()

    assert return_value == expected_response
    assert logger_mock.call_count == 2

    hello_request_mock.assert_called_once_with()
    instance.stub.Hello.called_once_with('request')
