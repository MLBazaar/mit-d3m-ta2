import argparse
import logging
import os
import time
from concurrent import futures

import grpc
from ta3ta2_api import core_pb2_grpc

from ta2.logging import logging_setup
from ta2.ta3 import core_servicer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

LOGGER = logging.getLogger(__name__)


def serve(port, cs, daemon=False):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    core_pb2_grpc.add_CoreServicer_to_server(cs, server)
    server.add_insecure_port('[::]:{}'.format(port))

    LOGGER.info("Starting TA2 server on port %s", port)
    server.start()

    if not daemon:
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)

        LOGGER.info("TA2 server stopped")

    else:
        return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TA3 API Server')
    parser.add_argument('--port', type=int, default=45042)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-d', '--input', nargs='?')
    parser.add_argument('-o', '--output', nargs='?')
    parser.add_argument('-T', '--timeout', type=int, nargs='?')
    parser.add_argument('-L', '--logfile', type=str, nargs='?')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    output_dir = args.output or os.getenv('D3MOUTPUTDIR', 'output')
    input_dir = args.input or os.getenv('D3MINPUTDIR', 'input')
    timeout = args.timeout or os.getenv('D3MTIMEOUT', 600)
    debug = args.debug

    try:
        timeout = int(timeout)
    except ValueError:
        # FIXME This is just to be sure that it does not crash
        timeout = 600

    logging_setup(args.verbose, args.logfile)
    logging.getLogger("d3m.metadata.pipeline_run").setLevel(logging.ERROR)

    cs = core_servicer.CoreServicer(input_dir, output_dir, timeout, debug)
    serve(args.port, cs)
