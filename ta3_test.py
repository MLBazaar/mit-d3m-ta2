import argparse
import logging
import os
import time
from threading import Thread

from ta2.ta3.server import serve
from ta2.ta3 import core_servicer
from ta2.ta3.client import TA3APIClient
from ta2.logging import logging_setup


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
    description = client.describe_solution(solution_id)

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
    produce_solutions = client.get_produce_solution_results(request_id)

    print('### {} => client.SolutionExport("{}")'.format(dataset, fitted_solution_id))
    client.solution_export(fitted_solution_id, 1)

    print('### {} => client.EndSearchSolutions("{}")'.format(dataset, search_id))
    client.end_search_solutions(search_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test TA3 API')
    parser.add_argument('--no-server', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--port', type=int, default=45042)
    parser.add_argument('-i', '--input', default='input', nargs='?')
    parser.add_argument('-o', '--output', default='output', nargs='?')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-l', '--logfile', type=str, nargs='?')
    parser.add_argument('-T', '--timeout', type=int, default=60)
    parser.add_argument('-L', '--server-logfile', default='logs/ta3_api_server.log', nargs='?')
    parser.add_argument('datasets', nargs='*')

    args = parser.parse_args()

    if not args.datasets:
        args.datasets = os.listdir(args.input)

    os.makedirs('logs', exist_ok=True)
    logging_setup(1 + args.verbose, args.server_logfile)
    logging_setup(1 + args.verbose, args.logfile, logger_name='mit_ta2.ta3_api.ta3_api_client')

    server = None
    if not args.no_server:
        cs = core_servicer.CoreServicer(
            args.input,
            args.output,
            args.timeout,
            args.debug
        )
        server = serve(args.port, cs, True)
        # server = Thread(target=serve, args=(args.port, cs, True))
        # server.start()
        time.sleep(1)

    client = TA3APIClient(args.port, args.input)

    print('### Hello ###')
    hello = client.hello()

    for dataset in args.datasets:
        run_test(dataset, args.timeout / 60)

    if server:
        server.stop(0)
