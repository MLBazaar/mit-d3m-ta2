from unittest.mock import MagicMock, patch

from ta3_test import run_test


@patch('ta3_test.print')
def test_run_test(print_mock):
    # -- test setup
    client_mock = MagicMock()

    # client.SearchSolutions
    search_response = MagicMock(search_id='search_id')
    client_mock.search_solutions = MagicMock(return_value=search_response)

    # client.GetSearchSolutionsResults
    solutions = [MagicMock(solution_id='solution_id')]
    client_mock.get_search_solutions_results = MagicMock(return_value=solutions)

    # client.StopSearchSolutions
    client_mock.stop_search_solutions = MagicMock()

    # client.DescribeSolution
    client_mock.describe_solution = MagicMock()

    # client.ScoreSolution
    score_response = MagicMock(request_id='request_id')
    client_mock.score_solution = MagicMock(return_value=score_response)

    # client.GetScoreSolutionsResults
    client_mock.get_score_solution_results = MagicMock()

    # client.FitSolution
    fit_response = MagicMock(request_id='request_id')
    client_mock.fit_solution = MagicMock(return_value=fit_response)

    # client.GetFitSolutionsResults
    fitted_solutions = [MagicMock(fitted_solution_id='fitted_solution_id')]
    client_mock.get_fit_solution_results = MagicMock(return_value=fitted_solutions)

    # client.ProduceSolution
    produce_response = MagicMock(request_id='request_id')
    client_mock.produce_solution = MagicMock(return_value=produce_response)

    # client.GetProduceSolutionsResults
    client_mock.get_produce_solution_results = MagicMock()

    # client.SolutionExport
    client_mock.solution_export = MagicMock()

    # client.EndSearchSolutions
    client_mock.end_search_solutions = MagicMock()

    dataset = {}
    timeout = 0.5

    # -- actual test
    result = run_test(client_mock, dataset, timeout)

    assert result is None
    assert print_mock.call_count == 13

    client_mock.search_solutions.assert_called_once_with(dataset, timeout)
    client_mock.get_search_solutions_results.assert_called_once_with('search_id', 2)
    client_mock.stop_search_solutions.assert_called_once_with('search_id')
    client_mock.describe_solution.assert_called_once_with('solution_id')
    client_mock.score_solution.assert_called_once_with('solution_id', dataset)
    client_mock.get_score_solution_results.assert_called_once_with('request_id')
    client_mock.fit_solution.assert_called_once_with('solution_id', dataset)
    client_mock.get_fit_solution_results.assert_called_once_with('request_id')
    client_mock.produce_solution.assert_called_once_with('fitted_solution_id', dataset)
    client_mock.get_produce_solution_results.assert_called_once_with('request_id')
    client_mock.solution_export.assert_called_once_with('fitted_solution_id', 1)
    client_mock.end_search_solutions.assert_called_once_with('search_id')
