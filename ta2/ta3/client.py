import logging
import os
import re
from datetime import datetime

import grpc
from d3m.metadata.problem import Problem
from google.protobuf.timestamp_pb2 import Timestamp
from ta3ta2_api import core_pb2, core_pb2_grpc
from ta3ta2_api.utils import encode_problem_description
from ta3ta2_api.value_pb2 import Value, ValueType

LOGGER = logging.getLogger(__name__)

RE_PYTHONIZE = re.compile(r'[A-Z]')


def pythonize(name):
    return re.sub(r'[A-Z]', r'_\g<0>', name).upper()


class TA3APIClient(object):

    def __init__(self, port, local_input='input', remote_input='input', verbose=False):
        channel = grpc.insecure_channel('localhost:' + str(port))
        self.stub = core_pb2_grpc.CoreStub(channel)
        self.local_input = local_input
        self.remote_input = remote_input
        self.verbose = verbose

    def _get_dataset_doc_path(self, dataset):
        return os.path.join(
            'file://' + os.path.abspath(self.remote_input),
            dataset,
            'TRAIN/dataset_TRAIN/datasetDoc.json'
        )

    def _get_problem_doc_path(self, dataset):
        return os.path.join(
            'file://' + os.path.abspath(self.local_input),
            dataset,
            'TRAIN/problem_TRAIN/problemDoc.json'
        )

    def _build_problem(self, dataset):
        problem = Problem.load(problem_uri=self._get_problem_doc_path(dataset))
        return encode_problem_description(problem)

    def search_solutions(self, dataset, time_bound_search=1.):

        created_at = Timestamp()
        created_at.FromDatetime(datetime.utcnow())

        request = core_pb2.SearchSolutionsRequest(
            user_agent='ta3_api_test.py',
            version='2019.12.4',
            time_bound_search=time_bound_search,
            priority=0.,
            allowed_value_types=[
                ValueType.Value('RAW'),
                ValueType.Value('DATASET_URI'),
                ValueType.Value('CSV_URI'),
            ],
            inputs=[
                Value(dataset_uri=self._get_dataset_doc_path(dataset))
            ],
            problem=self._build_problem(dataset)
            # template=pipeline_pb2.PipelineDescription(
            #     id='dummy',
            #     source=pipeline_pb2.PipelineSource(
            #         name='dummy',
            #         contact='dummy',
            #         pipelines=['dummy'],
            #     ),
            #     created=created_at,
            #     context=pipeline_pb2.PipelineContext.Value('PIPELINE_CONTEXT_UNKNOWN'),
            #     name='dummy',
            #     description='dummy',
            #     users=[
            #         pipeline_pb2.PipelineDescriptionUser(
            #             id='dummy',
            #             reason='dummy',
            #             rationale='dummy'
            #         )
            #     ],
            #     inputs=[
            #         pipeline_pb2.PipelineDescriptionInput(
            #             name='dummy'
            #         )
            #     ],
            #     outputs=[
            #         pipeline_pb2.PipelineDescriptionOutput(
            #             name='dummy',
            #             data='dummy'
            #         )
            #     ],
            #     steps=[
            #         pipeline_pb2.PipelineDescriptionStep(
            #             primitive=pipeline_pb2.PrimitivePipelineDescriptionStep(
            #                 primitive=Primitive(
            #                     id='dummy',
            #                     version='dummy',
            #                     python_path='dummy',
            #                     name='dummy',
            #                     digest='dummy'
            #                 ),
            #                 arguments={
            #                     'dummy': pipeline_pb2.PrimitiveStepArgument(
            #                         data=pipeline_pb2.DataArgument(
            #                             data='dummy'
            #                         )
            #                     )
            #                 },
            #                 outputs=[
            #                     pipeline_pb2.StepOutput(
            #                         id='dummy'
            #                     )
            #                 ],
            #                 hyperparams={
            #                     'dummy': pipeline_pb2.PrimitiveStepHyperparameter(
            #                         data=pipeline_pb2.DataArgument(
            #                             data='dummy'
            #                         )
            #                     )
            #                 },
            #                 users=[
            #                     pipeline_pb2.PipelineDescriptionUser(
            #                         id='dummy',
            #                         reason='dummy',
            #                         rationale='dummy'
            #                     )
            #                 ],
            #             )
            #         )
            #     ]
            # ),
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.SearchSolutions(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def get_search_solutions_results(self, search_id, max_results=None):

        request = core_pb2.GetSearchSolutionsResultsRequest(
            search_id=search_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        solutions = []
        for solution in self.stub.GetSearchSolutionsResults(request):
            LOGGER.debug("%s: %s", solution.__class__.__name__, solution)

            solutions.append(solution)

            if max_results and len(solutions) >= max_results:
                break

        return solutions

    def end_search_solutions(self, search_id):
        request = core_pb2.EndSearchSolutionsRequest(
            search_id=search_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.EndSearchSolutions(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def stop_search_solutions(self, search_id):

        request = core_pb2.StopSearchSolutionsRequest(
            search_id=search_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.StopSearchSolutions(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def describe_solution(self, solution_id):

        request = core_pb2.DescribeSolutionRequest(
            solution_id=solution_id
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.DescribeSolution(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def score_solution(self, solution_id, dataset):

        problem = self._build_problem(dataset)

        request = core_pb2.ScoreSolutionRequest(
            solution_id=solution_id,
            inputs=[
                Value(dataset_uri=self._get_dataset_doc_path(dataset))
            ],
            performance_metrics=problem.problem.performance_metrics,
            # users=[
            #     core_pb2.SolutionRunUser(
            #         id='dummy',
            #         choosen=True,
            #         reason='dummy'
            #     )
            # ],
            configuration=core_pb2.ScoringConfiguration(
                method=core_pb2.EvaluationMethod.Value('K_FOLD'),
                folds=3,
                train_test_ratio=3,
                shuffle=True,
                random_seed=0,
                stratified=False
            )
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.ScoreSolution(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def get_score_solution_results(self, request_id):

        request = core_pb2.GetScoreSolutionResultsRequest(
            request_id=request_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        score = None
        for score in self.stub.GetScoreSolutionResults(request):
            LOGGER.debug("%s: %s", score.__class__.__name__, score)

        # return only the last one
        return score

    def fit_solution(self, solution_id, dataset):

        request = core_pb2.FitSolutionRequest(
            solution_id=solution_id,
            inputs=[
                Value(dataset_uri=self._get_dataset_doc_path(dataset))
            ],
            expose_outputs=[
                'outputs.0'
                # 'steps.0.produce'
            ],
            expose_value_types=[
                ValueType.Value('CSV_URI')
            ],
            # users=[
            #     core_pb2.SolutionRunUser(
            #         id='dummy',
            #         choosen=True,
            #         reason='dummy'
            #     )
            # ]
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.FitSolution(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def get_fit_solution_results(self, request_id, max_results=None):

        request = core_pb2.GetFitSolutionResultsRequest(
            request_id=request_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        fitted_solutions = []
        for solution in self.stub.GetFitSolutionResults(request):
            LOGGER.debug("%s: %s", solution.__class__.__name__, solution)

            fitted_solutions.append(solution)

            if max_results and len(fitted_solutions) >= max_results:
                break

        return fitted_solutions

    def produce_solution(self, fitted_solution_id, dataset):

        request = core_pb2.ProduceSolutionRequest(
            fitted_solution_id=fitted_solution_id,
            inputs=[
                Value(dataset_uri=self._get_dataset_doc_path(dataset))
            ],
            expose_outputs=[
                'outputs.0'
            ],
            expose_value_types=[
                ValueType.Value('CSV_URI')
            ],
            # users=[
            #     core_pb2.SolutionRunUser(
            #         id='dummy',
            #         choosen=True,
            #         reason='dummy'
            #     )
            # ]
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.ProduceSolution(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def get_produce_solution_results(self, request_id):
        request = core_pb2.GetProduceSolutionResultsRequest(
            request_id=request_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        results = []
        for result in self.stub.GetProduceSolutionResults(request):
            LOGGER.debug("%s: %s", result.__class__.__name__, result)

        return results

    def solution_export(self, fitted_solution_id, rank):
        request = core_pb2.SolutionExportRequest(
            solution_id=fitted_solution_id,
            rank=rank,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.SolutionExport(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def update_problem(self, search_id):
        request = core_pb2.UpdateProblemRequest(
            search_id=search_id,
        )

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.UpdateProblem(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def list_primitives(self):
        request = core_pb2.ListPrimitivesRequest()

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.ListPrimitives(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response

    def hello(self):
        request = core_pb2.HelloRequest()

        LOGGER.debug("%s: %s", request.__class__.__name__, request)

        response = self.stub.Hello(request)

        LOGGER.debug("%s: %s", response.__class__.__name__, response)

        return response
