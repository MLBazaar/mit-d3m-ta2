import argparse
import logging
import os
import time

from ta2.ta3.server import serve
from ta2.ta3.client import TA3APIClient
from ta2 import logging_setup


def run_test(dataset, timeout):
    print('### Testing dataset {} ###'.format(dataset))

    print('### {} => client.SearchSolutions("{}")'.format(dataset, dataset))
    search_response = client.search_solutions(dataset, timeout)
    search_id = search_response.search_id

    print('### {} => client.GetSearchSolutionsResults("{}")'.format(dataset, search_id))
    solutions = client.get_search_solutions_results(search_id, 2)
    solution_id = solutions[-1].solution_id

    print('### {} => client.StopSearchSolutions("{}")'.format(dataset, solution_id))
    client.stop_search_solutions(search_id)

    print('### {} => client.DescribeSolution("{}")'.format(dataset, search_id))
    client.describe_solution(solution_id)

    print('### {} => client.ScoreSolution("{}")'.format(dataset, solution_id))
    score_response = client.score_solution(solution_id, dataset)
    request_id = score_response.request_id

    print('### {} => client.GetScoreSolutionsResults("{}")'.format(dataset, request_id))
    client.get_score_solution_results(request_id)

    print('### {} => client.FitSolution("{}")'.format(dataset, solution_id))
    fit_response = client.fit_solution(solution_id, dataset)
    request_id = fit_response.request_id

    print('### {} => client.GetFitSolutionsResults("{}")'.format(dataset, request_id))
    fitted_solutions = client.get_fit_solution_results(request_id)
    fitted_solution_id = fitted_solutions[-1].fitted_solution_id

    print('### {} => client.ProduceSolution("{}")'.format(dataset, fitted_solution_id))
    produce_response = client.produce_solution(fitted_solution_id, dataset)
    request_id = produce_response.request_id

    print('### {} => client.GetProduceSolutionsResults("{}")'.format(dataset, request_id))
    client.get_produce_solution_results(request_id)

    print('### {} => client.SolutionExport("{}")'.format(dataset, fitted_solution_id))
    client.solution_export(fitted_solution_id, 1)

    print('### {} => client.EndSearchSolutions("{}")'.format(dataset, search_id))
    client.end_search_solutions(search_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test TA3 API')
    parser.add_argument('--no-server', action='store_true', help=(
        'Do not start a server instance. Useful to start a separated instance for debug purposes'
    ))
    parser.add_argument('--debug', action='store_true',
                        help='Start the server in sync mode. Needed for debugging.')
    parser.add_argument('--port', type=int, default=45042,
                        help='Port to use, both for client and server.')
    parser.add_argument('-i', '--input', default='input',
                        help='Path to the datsets root folder')
    parser.add_argument('-o', '--output', default='output',
                        help='Path to the folder where outputs will be stored')
    parser.add_argument('-t', '--timeout', type=int,
                        help='Maximum time allowed for the tuning, in number of seconds')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity')
    parser.add_argument('-l', '--logfile', type=str,
                        help='Path to the logging file. If not given, log to stdout')
    parser.add_argument('-L', '--server-logfile', default='logs/ta3_api_server.log',
                        help='Path to the server logging file')
    parser.add_argument('datasets', nargs='*')

    args = parser.parse_args()

    if not args.datasets:
        args.datasets = os.listdir(args.input)

    os.makedirs('logs', exist_ok=True)
    logging_setup(1 + args.verbose, args.server_logfile)
    logging_setup(1 + args.verbose, args.logfile, logger_name='mit_ta2.ta3_api.ta3_api_client')
    logging.getLogger("d3m.metadata.pipeline_run").setLevel(logging.ERROR)

    server = None
    if not args.no_server:
        server = serve(
            args.port,
            args.input,
            args.output,
            args.timeout,
            args.debug,
            True
        )
        time.sleep(1)

    client = TA3APIClient(args.port, args.input)

    print('### Hello ###')
    hello = client.hello()

    for dataset in args.datasets:
        run_test(dataset, args.timeout / 60)

    if server:
        server.stop(0)
