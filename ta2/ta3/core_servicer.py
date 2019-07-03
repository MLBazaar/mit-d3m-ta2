import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse

from d3m.container.dataset import Dataset
from d3m.metadata.base import Context
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import Problem
from d3m.runtime import Runtime
from google.protobuf.timestamp_pb2 import Timestamp
from ta3ta2_api import core_pb2, core_pb2_grpc, pipeline_pb2, primitive_pb2, problem_pb2, value_pb2
from ta3ta2_api.utils import decode_performance_metric, decode_problem_description, encode_score

from ta2.search import PipelineSearcher
from ta2.utils import dump_pipeline


def recursivedict():
    return defaultdict(recursivedict)


def camel_case(name):
    parts = name.split('_')
    out = parts[0]

    for part in parts[1:]:
        out += part[0].upper() + part[1:]

    return out


LOGGER = logging.getLogger(__name__)

VERSION = core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version]


"""
CORE SHARED OBJECTS
===================

enum EvaluationMethod {
    // Default value. Not to be used.
    EVALUATION_METHOD_UNDEFINED = 0;

    // The following are the only evaluation methods required
    // to be supported for the "ScoreSolution" call.
    HOLDOUT = 1;
    K_FOLD = 2;

    // The rest are defined to allow expressing internal evaluation
    // methods used by TA2 during solution search. If any method being used
    // is missing, feel free to request it to be added.
    LEAVE_ONE_OUT = 100;
    // Instead of really scoring, a TA2 might predict the score only.
    PREDICTION = 101;
    // Training data is reused to test as well.
    TRAINING_DATA = 102;
}

message ScoringConfiguration {
    // The evaluation method to use.
    EvaluationMethod method = 1;
    // Number of folds made, if applicable.
    int32 folds = 2;
    // Ratio of train set vs. test set, if applicable.
    double train_test_ratio = 3;
    // Shuffle data? Set to true if employed.
    bool shuffle = 4;
    // Value for random seed to use for shuffling. Optional.
    int32 random_seed = 5;
    // Do stratified k-fold? Set to true if employed.
    bool stratified = 6;
}

message Score {
    ProblemPerformanceMetric metric = 1;
    // When doing multiple folds, which fold is this score associated with, 0-based.
    // We do not aggregate scores across folds on the TA2 side, but expose everything to the TA3.
    // If scoring was not done as part of the cross-validation, then it can be returned
    // as the first and only fold, in which case the value of this field should be 0.
    int32 fold = 2;
    // To which target or targets does this score apply?
    repeated ProblemTarget targets = 3;
    Value value = 4;
}

enum ProgressState {
    // Default value. Not to be used.
    PROGRESS_UNKNOWN = 0;

    // The process has been scheduled but is pending execution.
    PENDING = 1;
    // The process is currently running. There can be multiple messages with this state
    // (while the process is running).
    RUNNING = 2;
    // The process completed and final results are available.
    COMPLETED = 3;
    // The process failed.
    ERRORED = 4;
}

// After "state" becomes "COMPLETED" or "ERRORED" stream closes.
// The granularity of progress updates is not specified by the API at this time. Some systems
// might be updating frequently and provide many updates of the progress of a whole process
// as well as individual pipeline steps. Some systems might just report these high-level
// progress states once, not doing any progress updates in the meantime.  The "status" field
// should contain information to supplement the progress state, such as specific failure details
// in the case of an "ERRORED" state being returned.
message Progress {
    ProgressState state = 1;
    string status = 2;
    // Set only after state becomes "RUNNING". If it never really properly runs, but errors
    // when attempted to run, then it should be the timestamp of the error.
    google.protobuf.Timestamp start = 3;
    // Set only when state is "COMPLETED" or "ERRORED".
    google.protobuf.Timestamp end = 4;
}

// Description of a TA2 score done during solution search. Because there is a wide range of
// potential approaches a TA2 can use to score candidate solutions this might not capture what
//  your TA2 is doing. Feel free to request additions to be able to describe your approach.
message SolutionSearchScore {
    ScoringConfiguration scoring_configuration = 1;
    repeated Score scores = 2;
}

message PrimitiveStepDescription {
    // Selected value for free pipeline hyper-parameters.
    map<string, Value> hyperparams = 1;
}

message SubpipelineStepDescription {
    // Each step in a sub-pipeline has a description. These are reported in the order of steps
    // in the sub-pipeline.
    repeated StepDescription steps = 1;
}

message StepDescription {
    oneof step {
        PrimitiveStepDescription primitive = 1;
        SubpipelineStepDescription pipeline = 2;
    }
}

message StepProgress {
    Progress progress = 1;
    // If step is a sub-pipeline, then this list contains progress for each step in the
    // sub-pipeline, in order.
    // List can be incomplete while the process is in progress. Systems can provide
    // steps only at the end (when "progress" equals COMPLETED) and not during running.
    repeated StepProgress steps = 2;
}

// User associated with the run of the solution.
message SolutionRunUser {
    // A UUID of the user. It does not have to map to any real ID, just that it is possible
    // to connect multiple solution actions by the same user together, if necessary.
    string id = 1;
    // Was this run because solution was choosen by this user.
    bool choosen = 2;
    // Textual reason provided by the user why the run was choosen by this user.
    string reason = 3;
}


RPC SPECIFICATION
=================

// See each message's comments for information about each particular call.
service Core {
    rpc SearchSolutions (SearchSolutionsRequest) returns (SearchSolutionsResponse) {}
    rpc GetSearchSolutionsResults (GetSearchSolutionsResultsRequest) returns (stream GetSearchSolutionsResultsResponse) {}
    rpc EndSearchSolutions (EndSearchSolutionsRequest) returns (EndSearchSolutionsResponse) {}
    rpc StopSearchSolutions (StopSearchSolutionsRequest) returns (StopSearchSolutionsResponse) {}

    rpc DescribeSolution (DescribeSolutionRequest) returns (DescribeSolutionResponse) {}

    rpc ScoreSolution (ScoreSolutionRequest) returns (ScoreSolutionResponse) {}
    rpc GetScoreSolutionResults (GetScoreSolutionResultsRequest) returns (stream GetScoreSolutionResultsResponse) {}

    rpc FitSolution (FitSolutionRequest) returns (FitSolutionResponse) {}
    rpc GetFitSolutionResults (GetFitSolutionResultsRequest) returns (stream GetFitSolutionResultsResponse) {}

    rpc ProduceSolution (ProduceSolutionRequest) returns (ProduceSolutionResponse) {}
    rpc GetProduceSolutionResults (GetProduceSolutionResultsRequest) returns (stream GetProduceSolutionResultsResponse) {}

    rpc SolutionExport (SolutionExportRequest) returns (SolutionExportResponse) {}

    rpc UpdateProblem (UpdateProblemRequest) returns (UpdateProblemResponse) {}

    rpc ListPrimitives (ListPrimitivesRequest) returns (ListPrimitivesResponse) {}

    rpc Hello (HelloRequest) returns (HelloResponse) {}
}


VALUE PROTO OBJECTS
===================

// All values are immutable and no files should be changed after a URI
// is provided to the other system. When using shared file system, all
// URIs should be absolute to the file system, for example
// "file:///datasets/dataset_1/datasetDoc.json". It is assumed that both
// TA2 and TA3 systems both have a limited number of shared directories
// mounted at same locations (in previous example, "/datasets" directory).
// When one system creates a dataset and sends over the URI, the other can
// directly access it without doing any extra work (like downloading or copying).
//
// Configuration of shared directories and shared instance of Plasma are not
// specified by this API.
//
// Not all types of non-raw values is necessary to be supported/allowed.
// Both systems maintain a list of allowed value types the other system accepts.
// Some calls also provide a way to provide such a list. When a value is to be
// provided to the other system, the list is traversed in order and the first
// value type which can be used without an error is used. If the list is
// exhausted, then an error is provided instead.

enum ValueType {
    // Default value. Not to be used.
    VALUE_TYPE_UNDEFINED = 0;

    // The following value types are those everyone should support.

    // Raw value. Not all values can be represented as a raw value.
    RAW = 1;
    // Represent the value as a D3M dataset. Only "file://" schema is supported using a
    // shared file system. Dataset URI should point to the "datasetDoc.json" file of the dataset.
    // Only Dataset container values can be represented this way.
    DATASET_URI = 2;
    // Represent the value as a CSV file. Only "file://" schema is supported using a
    // shared file system. CSV URI should point to the file with ".csv" file extension.
    // Only tabular container values with numberic and string cell values can be represented
    // this way.
    CSV_URI = 3;

    // The following are additional value types which can be supported by systems,
    // but it is not required. If the value cannot be represented with value types your system
    // supports and your system is still asked to do so, it should return "ValueError" error instead.

    // Represent values by Python-pickling them. Only "file://" schema is supported using a
    // shared file system. Pickle URI should point to the file with ".pickle" file extension.
    PICKLE_URI = 4;
    // Represent values by Python-pickling them but sending them through the API.
    PICKLE_BLOB = 5;
    // Represent values with arrow and storing them into shared instance of Plasma.
    PLASMA_ID = 6;
}

message ValueError {
    // A error message useful for debugging or logging. Not meant to be very end-user friendly.
    // If a list of supported/allowed value types could not support a given value, then message
    // should say so. On the other hand, if there was really an error using a value type which
    // would otherwise support a given value, then the error message should communicate this error.
    // If there was such an error but some later value type allowed for recovery, then there
    // should be no error.
    string message = 1;
}

message DoubleList {
    repeated double list = 1;
}

message Int64List {
    repeated int64 list = 1;
}

message BoolList {
    repeated bool list = 1;
}

message StringList {
    repeated string list = 1;
}

message BytesList {
    repeated bytes list = 1;
}

message Value {
    oneof value {
        // If there was an error trying to provided the value using the requested
        // value type and no other value type was available to be used.
        ValueError error = 1;
        // Raw values directly provided in the message.
        double double = 2;
        int64 int64 = 3;
        bool bool = 4;
        string string = 5;
        bytes bytes = 6;
        DoubleList double_list = 7;
        Int64List int64_list = 8;
        BoolList bool_list = 9;
        StringList string_list = 10;
        BytesList bytes_list = 11;
        // An URI pointing to a dataset. Resulting value is Dataset container value from loading this URI.
        string dataset_uri = 12;
        // An URI pointing to a CSV file.
        string csv_uri = 13;
        // An URI to a Python-pickled value.
        string pickle_uri = 14;
        // A Python-pickled value itself.
        bytes pickle_blob = 15;
        // 20 bytes of Plasma ObjectID of the value.
        bytes plasma_id = 16;
    }
}


PROBLEM PROTO OBJECTS
=====================

// Top level classification of the problem.
enum TaskType {
    // Default value. Not to be used.
    TASK_TYPE_UNDEFINED = 0;

    CLASSIFICATION = 1;
    REGRESSION = 2;
    CLUSTERING = 3;
    LINK_PREDICTION = 4;
    VERTEX_NOMINATION = 5;
    COMMUNITY_DETECTION = 6;
    GRAPH_CLUSTERING = 7;
    GRAPH_MATCHING = 8;
    TIME_SERIES_FORECASTING = 9;
    COLLABORATIVE_FILTERING = 10;
    OBJECT_DETECTION = 11;
}

// Secondary classification of the problem.
enum TaskSubtype {
    // Default value. Not to be used.
    TASK_SUBTYPE_UNDEFINED = 0;

    // No secondary task is applicable for this problem.
    NONE = 1;
    BINARY = 2;
    MULTICLASS = 3;
    MULTILABEL = 4;
    UNIVARIATE = 5;
    MULTIVARIATE = 6;
    OVERLAPPING = 7;
    NONOVERLAPPING = 8;
}

// The evaluation metric for any potential solution.
enum PerformanceMetric {
    // Default value. Not to be used.
    METRIC_UNDEFINED = 0;

    // The following are the only evaluation methods required
    // to be supported for the ScoreSolution call.
    ACCURACY = 1;
    PRECISION = 2;
    RECALL = 3;
    F1 = 4;
    F1_MICRO = 5;
    F1_MACRO = 6;
    ROC_AUC = 7;
    ROC_AUC_MICRO = 8;
    ROC_AUC_MACRO = 9;
    MEAN_SQUARED_ERROR = 10;
    ROOT_MEAN_SQUARED_ERROR = 11;
    ROOT_MEAN_SQUARED_ERROR_AVG = 12;
    MEAN_ABSOLUTE_ERROR = 13;
    R_SQUARED = 14;
    NORMALIZED_MUTUAL_INFORMATION = 15;
    JACCARD_SIMILARITY_SCORE = 16;
    PRECISION_AT_TOP_K = 17;
    OBJECT_DETECTION_AVERAGE_PRECISION = 18;

    // The rest are defined to allow expressing internal evaluation
    // scores used by TA2 during pipeline search. If any you are using
    // is missing, feel free to request it to be added.
    // Average loss of an unspecified loss function.
    LOSS = 100;
}

message ProblemPerformanceMetric {
    PerformanceMetric metric = 1;
    // Additional params used by some metrics.
    int32 k = 2;
    string pos_label = 3;
}

message Problem {
    // ID of this problem.
    string id = 1;
    // Version of this problem.
    string version = 2;
    string name = 3;
    string description = 4;
    TaskType task_type = 5;
    TaskSubtype task_subtype = 6;
    repeated ProblemPerformanceMetric performance_metrics = 7;
}

message ProblemTarget {
    int32 target_index = 1;
    string resource_id = 2;
    int32 column_index = 3;
    string column_name = 4;
    int32 clusters_number = 5;
}

message ProblemInput {
    // Should match one of input datasets given to the pipeline search.
    // Every "Dataset" object has an "id" associated with it and is available
    // in its metadata. That ID is then used here to reference those inputs.
    string dataset_id = 1;
    // Targets should resolve to columns in a given dataset.
    repeated ProblemTarget targets = 2;
}

// Problem description matches the parsed problem description by
// the d3m_metadata.problem.Problem.load Python method.
// Problem outputs are not necessary for the purpose of this API
// and are needed only when executing an exported pipeline, but then
// TA2 gets full problem description anyway directly.
message ProblemDescription {
    Problem problem = 1;
    repeated ProblemInput inputs = 2;
}

PIPELINE PROTO OBJECTS
======================

// Pipeline description contains many "data references". Data reference is just a string
// which identifies an output of a step or a pipeline input and forms a data-flow connection
// between data available and an input to a step. It is recommended to be a string of the
// following forms:
//
//  * `steps.<number>.<id>` — `number` identifies the step in the list of steps (0-based)
//    and `id` identifies the name of a produce method of the primitive,
//    or the output of a pipeline step
//
//  * `inputs.<number>` — `number` identifies the pipeline input (0-based)
//
//  * `outputs.<number>` — `number` identifies the pipeline output (0-based)

message ContainerArgument {
    // Data reference.
    string data = 1;
}

message DataArgument {
    // Data reference.
    string data = 1;
}

message DataArguments {
    repeated string data = 1;
}

message PrimitiveArgument {
    // 0-based index identifying a step of which primitive is used as a value.
    int32 data = 1;
}

message PrimitiveArguments {
    // 0-based index identifying a step of which primitive is used as a value.
    repeated int32 data = 1;
}

message ValueArgument {
    Value data = 1;
}

message PrimitiveStepArgument {
    oneof argument {
        // A container data type as an argument.
        ContainerArgument container = 1;
        // A singleton output from another step as an argument.
        DataArgument data = 2;
    }
}

message PrimitiveStepHyperparameter {
    oneof argument {
        // A container data type as a hyper-parameter.
        ContainerArgument container = 1;
        // A singleton output from another step as a hyper-parameter.
        DataArgument data = 2;
        // A primitive instance to be passed as a hyper-parameter.
        PrimitiveArgument primitive = 3;
        // A constant value of a hyper-parameter.
        ValueArgument value = 4;
        // "A set of singleton outputs from other steps in a pipeline.
        DataArguments data_set = 5;
        // A set of primitive instances to be passed as a hyper-parameter.
        PrimitiveArguments primitives_set = 6;
    }
}

message StepInput {
    // Data reference.
    string data = 1;
}

message StepOutput {
    // Name which becomes part of the data reference.
    string id = 1;
}

message PipelineSource {
    // String representing name of the author, team.
    string name = 1;
    // An URI to contact the source.
    string contact = 2;
    // A list of pipeline IDs used to derive the pipeline.
    repeated string pipelines = 3;
}

enum PipelineContext {
    // Default value. Not to be used.
    PIPELINE_CONTEXT_UNKNOWN = 0;

    // Pipeline was created during building/training of the system itself, e.g., during metalearning.
    PRETRAINING = 1;
    // Pipeline was created during development or testing of the system itself, e.g., during debugging.
    TESTING = 2;
    // Pipeline was created during evaluation of the system itself, e.g., NIST blind evaluation.
    EVALUATION = 3;
    // Pipeline was created during regular (production) operation of the system.
    PRODUCTION = 4;
}

// User associated with the creation of the template/pipeline, or selection of a primitive.
message PipelineDescriptionUser {
    // Globally unique ID for this user. It can be opaque, but it should identify the same user
    // across sessions. Consider using UUID variant 5 with namespace set to the name of your system
    // and name to an ID in your system's database. It does not have to map to any real ID, just
    // that it is possible to connect mutliple pipelines/templates by the same user together,
    // if necessary.
    string id = 1;
    // A natural language description of what the user did to be on the list, e.g., "Picked
    // a pipeline from a list of pipelines.".
    string reason = 2;
    // A natural language description by the user of what the user did,
    // e.g., "I picked a pipeline because it looks short in comparison with others.".
    string rationale = 3;
}

// Possible input to the pipeline or template.
message PipelineDescriptionInput {
    // Human friendly name of the input.
    string name = 1;
}

// Available output of the pipeline or template.
message PipelineDescriptionOutput {
    // Human friendly name of the output.
    string name = 1;
    // Data reference, probably of an output of a step.
    string data = 2;
}

message PrimitivePipelineDescriptionStep {
    Primitive primitive = 1;
    // Arguments to the primitive. Constructor arguments should not be listed here, because they
    // can be automatically created from other information. All these arguments are listed as kind
    // "PIPELINE" in primitive's metadata.
    map<string, PrimitiveStepArgument> arguments = 2;
    // List of produce metods providing data. One can reference using data reference these outputs
    // then in arguments (inputs) in other steps or pipeline outputs.
    repeated StepOutput outputs = 3;
    // Some hyper-parameters are not really tunable and should be fixed as part of template/pipeline.
    // This can be done here. Hyper-parameters listed here cannot be tuned or overridden. Author of a
    // template/pipeline decides which hyper-parameter are which, probably based on their semantic type.
    // TA3 can specify a list of hyper-parameters to fix, and TA2 can add to the list additional
    // hyper-paramaters in found pipelines.
    map<string, PrimitiveStepHyperparameter> hyperparams = 4;
    // List of users associated with selection of this primitive/arguments/hyper-parameters. Optional.
    repeated PipelineDescriptionUser users = 5;
}

message SubpipelinePipelineDescriptionStep {
    // Only "id" field is required in this case to reference another pipeline in the template.
    PipelineDescription pipeline = 1;
    // List of data references, probably of an output of a step or pipeline input,
    // mapped to sub-pipeline's inputs in order.
    repeated StepInput inputs = 2;
    // List of IDs to be used in data references, mapping sub-pipeline's outputs in order.
    repeated StepOutput outputs = 3;
}

// Used to represent a pipeline template which can be used to generate full pipelines.
// A placeholder is replaced with a pipeline step to form a pipeline. See README.md
// for restrictions on the number of them, their position, allowed inputs and outputs,
// etc.
message PlaceholderPipelineDescriptionStep {
    // List of inputs which can be used as inputs to resulting sub-pipeline. Resulting
    // sub-pipeline does not have to use all the inputs, but it cannot use any other inputs.
    repeated StepInput inputs = 1;
    // A list of outputs of the resulting sub-pipeline.
    repeated StepOutput outputs = 2;
}

message PipelineDescriptionStep {
    oneof step {
        PrimitivePipelineDescriptionStep primitive = 1;
        SubpipelinePipelineDescriptionStep pipeline = 2;
        PlaceholderPipelineDescriptionStep placeholder = 3;
    }
}

// Pipeline description matches the D3M pipeline description.
// It serves two purposes: describing found pipelines by TA2 to TA3, and communicating pipeline
// templates by TA3 to TA2. Because of this some fields are reasonable only in one of those uses.
// They are marked with "TA2" or "TA3" in the comment, for fields which are primarily to be set
// only by TA2 or only by TA3, respectivelly.
message PipelineDescription {
    // TA2: UUID of the pipeline. Templates do not have IDs. But TA3 might provide it for a fully
    // specified pipeline. It does not necessary have to match "solution_id" from
    // "ListSolutionsResponse" and other related messages. Those IDs are about whole solutions
    // (pipeline, potentially fitted, with set hyper-parameters). This here ID is about this
    // particular ID description.
    string id = 1;
    // "schema" field is not needed because it is fixed by the TA2-TA3 protocol version.
    // System which generated a pipeline or a template. Optional.
    PipelineSource source = 2;
    // TA2: Timestamp when created. Templates do not have this timestamp. TA3 might provide it for
    // a fully specified pipeline.
    google.protobuf.Timestamp created = 3;
    // In which context a template or pipeline was made. This is helpful to distinguish evaluation
    // context from other contexts. The value should not really influence different behavior from
    // either system, but it is useful when recording metalearning information to understand this.
    PipelineContext context = 4;
    // Human friendly name of the pipeline. For templates it can be a hint to
    // TA2 how to name found pipelines. Optional.
    string name = 5;
    // Human friendly description of the pipeline. Optional.
    string description = 6;
    // List of users associated with the creation of the template and consequently of the pipeline.
    // TA2 can store this information into metalearning database. TA2 is not really expected to use
    // this information during pipeline search. TA2 should not have to understand TA3 users, mapping
    // between users and pipeline search IDs is something TA3 should handle. Optional.
    repeated PipelineDescriptionUser users = 7;
    // In most cases inputs are datasets. But if TA3 wants to jut run a primitive, it can send a
    // template with only that primitive in the template, and then pass anything to its inputs during
    // execution. Here, we are describing possible inputs to the pipeline or template. Order matters.
    repeated PipelineDescriptionInput inputs = 8;
    // Available outputs of the pipeline or template.
    repeated PipelineDescriptionOutput outputs = 9;
    // Steps defining the pipeline.
    repeated PipelineDescriptionStep steps = 10;
}
"""


def dt2ts(dt):
    if dt is not None:
        ts = Timestamp()
        ts.FromDatetime(dt)
        return ts

    return None


class CoreServicer(core_pb2_grpc.CoreServicer):

    DB = recursivedict()

    def __init__(self, input_dir, output_dir, timeout, static, debug=False):

        super(CoreServicer, self).__init__()

        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.static = os.path.abspath(static)
        self.ranked_dir = os.path.join(self.output_dir, 'pipelines_ranked')
        self.timeout = timeout
        self.debug = debug

    def _build_problem(self, problem_description):
        # TODO: it might be removed, it's not being used.
        problem = problem_description.problem
        inputs = problem_description.inputs

        problem_dict = {
            "about": {
                "problemID": problem.id,
                "problemName": problem.name,
                "taskType": problem_pb2.TaskType.Name(problem.task_type).lower(),
                "taskSubType": problem_pb2.TaskSubtype.Name(problem.task_subtype),
                "problemVersion": problem.version,
                "problemSchemaVersion": "3.0"
            },
            "inputs": {
                "data": [
                    {
                        "datasetID": problem_input.dataset_id,
                        "targets": [
                            {
                                "targetIndex": target.target_index,
                                "resID": target.resource_id,
                                "colIndex": target.column_index,
                                "colName": target.column_name
                            }
                            for target in problem_input.targets
                        ]
                    }
                    for problem_input in inputs
                ],
                "performanceMetrics": [
                    {
                        "metric": camel_case(
                            problem_pb2.PerformanceMetric.Name(metric.metric).lower()
                        )
                    }
                    for metric in problem.performance_metrics
                ]
            },
            "expectedOutputs": {
                "predictionsFile": "predictions.csv"
            }
        }
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp_file:
            json.dump(problem_dict, tmp_file)

        return Problem.load(problem_uri='file://' + os.path.abspath(tmp_file.name))

    def _run_session(self, session, method, *args, **kwargs):
        exception = None
        try:
            method(*args, **kwargs)

        except Exception as e:
            LOGGER.exception("Exception in %s session %s", session['type'], session['id'])
            exception = e

        finally:
            LOGGER.info('Ending %s session %s', session['type'], session['id'])
            session['end'] = datetime.utcnow()
            session['done'] = True
            if exception is not None:
                session['error'] = '{}: {}'.format(exception.__class__.__name__, exception)

    def _start_session(self, session_id, session_type, method, *args, **kwargs):
        session = {
            'id': session_id,
            'type': session_type,
            'start': datetime.utcnow()
        }
        session.update(kwargs)

        self.DB[session_type + '_sessions'][session_id] = session

        args = [session, method] + list(args)

        sync_ = 'sync' if self.debug else 'async'
        LOGGER.info('Starting %s %s session %s', sync_, session['type'], session['id'])

        if self.debug:
            self._run_session(*args)

        else:
            threading.Thread(target=self._run_session, args=args).start()

    def SearchSolutions(self, request, context):
        LOGGER.info("\n######## SearchSolutions ########\n%s########", request)
        """
        rpc SearchSolutions (SearchSolutionsRequest) returns (SearchSolutionsResponse) {}

        // Starts a new solution search. Found solutions have not necessary been fitted on the provided
        // inputs. Problem description and inputs are used only to help guide the search process.
        // Consider any found solutions to be just a static description of solutions at this stage.
        // Multiple parallel solution searches can happen at the same time.
        message SearchSolutionsRequest {
            // Some string identifying the name and version of the TA3 system.
            string user_agent = 1;
            // Shall be set to "protocol_version" above.
            string version = 2;
            // Desired upper limit of time for solution search, expressed in minutes.
            // Is suggestion, and TA2's should attempt to obey, but TA3's should realize may be
            // violated. Default value of 0 (and any negative number) signifies no time bound.
            double time_bound = 3;
            // Value stating the priority of the search. If multiple searches are queued then highest
            // priority (largest number) should be started next by TA2. Primarily used to sort any
            // queue, but no further guarantee that TA2 can give more resources to high priority
            // searches. If unspecified, by default search will have priority 0. Negative numbers have
            // still lower priority.
            double priority = 4;
            // Which value types can a TA2 system use to communicate values to a TA3 system?
            // The order is important as a TA2 system will try value types in order until one works out,
            // or an error will be returned instead of the value.
            repeated ValueType allowed_value_types = 5;
            // Problem description to use for the solution search.
            ProblemDescription problem = 6;
            // A pipeline template to use for search or to execute. If template is omitted, then a
            // regular solution search is done. If template consists only of one placeholder step,
            // then a regular solution search is done to replace that step. If there is no placeholder
            // step, but template describes a full pipeline with free hyper-parameters, then this
            // call becomes a hyper-paramater tuning call over free hyper-paramaters and found solutions
            // share the same pipeline, but different hyper-parameter configurations. If there is no
            // placeholder step and all hyper-parameters are fixed as part of the pipeline, then this
            // call only checks the given template and returns the solution with same pipeline back, to
            // be executed. This allows fixed computations to be done on data, for example, pipeline can
            // consist of only one primitive with fixed hyper-parameters to execute that one primitive.
            // Moreover, such fully specified pipelines with fixed hyper-parametres can have any
            // inputs and any outputs. Otherwise pipelines have to be from a Dataset container value
            // to predictions Pandas dataframe. While there are all these options possible, only a
            // subset has to be supported by all systems. See README for more details.
            PipelineDescription template = 7;
            // Pipeline inputs used during solution search. They have to point to Dataset container
            // values. Order matters as each input is mapped to a template's input in order. Optional
            // for templates without a placeholder and with all hyper-parameters fixed.
            repeated Value inputs = 8;
        }

        // Call returns immediately with the ID. Use "GetFoundSolutions" call to get results.
        message SearchSolutionsResponse {
            // An ID identifying this solution search. This string should be at least 22 characters
            // long to ensure enough entropy to not be guessable.
            string search_id = 1;
        }
        """
        version = request.version
        time_bound_search = request.time_bound_search
        problem_description = request.problem
        inputs = request.inputs
        allowed_value_types = request.allowed_value_types

        # Ignored:
        # user_agent = request.user_agent
        # priority = request.priority
        # template = request.template

        # Validate input
        assert version == VERSION, 'Only version {} is supported'.format(VERSION)
        problem_inputs = len(problem_description.inputs)
        assert problem_inputs == 1 and len(inputs) == 1, 'Only one input is supported'

        search_id = str(uuid.uuid4())

        if time_bound_search:
            timeout = int(time_bound_search * 60)
        else:
            timeout = self.timeout

        problem = decode_problem_description(problem_description)

        searcher = PipelineSearcher(self.input_dir, self.output_dir)

        self._start_session(
            search_id,
            'search',
            searcher.search,
            problem,
            timeout,
            searcher=searcher,
            problem=problem,
            allowed_value_types=allowed_value_types
        )

        return core_pb2.SearchSolutionsResponse(
            search_id=search_id
        )

    def _get_progress(self, session):
        error = session.get('error')
        if error is not None:
            state = 'ERRORED'
        elif session.get('done'):
            state = 'COMPLETED'
        else:
            state = 'RUNNING'

        return core_pb2.Progress(
            state=core_pb2.ProgressState.Value(state),
            status=error,
            start=dt2ts(session.get('start')),
            end=dt2ts(session.get('end'))
        )

    def _stream(self, session, get_next, close_on_done=False):
        returned = 0
        stop = False
        while not stop:
            done = session.get('done')
            response = get_next(session, returned)

            if done and (close_on_done or not response):
                LOGGER.info("Closing stream")
                stop = True

            if response:
                returned += 1
                yield response

            else:
                time.sleep(1)

    def _get_search_soltuion_results(self, session, returned):
        solutions = session['searcher'].solutions

        if len(solutions) > returned:
            solution = solutions[returned]
            solution['session'] = session
            self.DB['solutions'][solution['id']] = solution

            return core_pb2.GetSearchSolutionsResultsResponse(
                progress=self._get_progress(session),
                done_ticks=returned,
                all_ticks=0.,
                solution_id=solution['id'],
                internal_score=solution['score'],
                scores=[
                    core_pb2.SolutionSearchScore(
                        # scoring_configuration=core_pb2.ScoringConfiguration(
                        #     method=core_pb2.EvaluationMethod.Value('K_FOLD'),
                        #     folds=5,
                        #     train_test_ratio=0.,
                        #     shuffle=True,
                        #     random_seed=0,
                        #     stratified=False
                        # ),
                        scores=[
                            core_pb2.Score(
                                metric=problem_pb2.ProblemPerformanceMetric(
                                    metric=problem_pb2.PerformanceMetric.Value('RANK')
                                ),
                                fold=0,
                                value=value_pb2.Value(
                                    raw=value_pb2.ValueRaw(double=solution['rank'])
                                )
                            )
                        ]
                    ),
                    # core_pb2.SolutionSearchScore(
                    #     scoring_configuration=core_pb2.ScoringConfiguration(
                    #         method=core_pb2.EvaluationMethod.Value('K_FOLD'),
                    #         folds=5,
                    #         train_test_ratio=0.,
                    #         shuffle=True,
                    #         random_seed=0,
                    #         stratified=False
                    #     ),
                    #     scores=[
                    #         core_pb2.Score(
                    #             fold=0,
                    #             targets=[
                    #                 ProblemTarget(
                    #                     target_index=0,
                    #                     resource_id="0",
                    #                     # column_index=0,
                    #                     # column_name="dummy",
                    #                     # clusters_number=0
                    #                 )
                    #             ],
                    #             value=value_pb2.Value(int64=1)
                    #         )
                    #     ]
                    # )
                ]
            )

    def GetSearchSolutionsResults(self, request, context):
        LOGGER.info("\n######## GetSearchSolutionsResults ########\n%s########", request)
        """
        rpc GetSearchSolutionsResults (GetSearchSolutionsResultsRequest) returns (stream GetSearchSolutionsResultsResponse) {}

        // Get all solutions presently identified by the search and start receiving any
        // further solutions also found as well.
        message GetSearchSolutionsResultsRequest {
            string search_id = 1;
        }

        message GetSearchSolutionsResultsResponse {
            // Overall process progress, not progress per solution. While solutions are being found and
            // returned, or scores computed and updated, progress state should be kept at "RUNNING".
            Progress progress = 1;
            // A measure of progress during search. It can be any number of internal steps or
            // actions a TA2 is doing during search. It can be even number of how many candidate
            // solutions were already examined. It does not even have to be an integer.
            // How regularly a change to this number is reported to TA3 is left to TA2's discretion,
            // but a rule of thumb is at least once a minute if the number changes.
            double done_ticks = 2;
            // If TA2 knows how many internal steps or actions are there, it can set this field.
            // This can also be updated through time if more (or even less) internal steps or
            // actions are determined to be necessary. If this value is non-zero, then it should
            // always hold that "done_ticks" <= "all_ticks".
            double all_ticks = 3;
            string solution_id = 4;
            // Internal score for this solution between 0.0 and 1.0 where 1.0 is the highest score.
            // There is no other meaning to this score and it does not necessary depend on scores
            // listed in the problem description. Optional.
            // Because this field is optional, if omitted the default value will be 0. But 0 is a
            // valid value for this field. Because of that you should never omit the field.
            // If you do not have internal score to provide, use NaN for the value of this field
            // to signal that.
            double internal_score = 5;
            // TA2 might be able to provide more meaningful scores as well, depending on its
            // approach to solution search. Moreover, even the same TA2 might not use the same scoring
            // approach for all of its solutions. Optional.
            repeated SolutionSearchScore scores = 6;
        }
        """
        search_id = request.search_id

        if search_id not in self.DB['search_sessions']:
            raise ValueError('Invalid search_id')

        session = self.DB['search_sessions'][search_id]

        return self._stream(session, self._get_search_soltuion_results)

    def EndSearchSolutions(self, request, context):
        LOGGER.info("\n######## EndSearchSolutions ########\n%s########", request)
        """
        rpc EndSearchSolutions (EndSearchSolutionsRequest) returns (EndSearchSolutionsResponse) {}

        // Ends the search and releases all resources associated with the solution search.
        // If the call is made in parallel with a running search and results are being streamed,
        // the search is stopped and the "GetSearchSolutionsResults" stream is closed by TA2
        // (as happens when the search is concluded on its own, or when a search is stopped
        // by "StopSearchSolutions"). Found solution IDs during the search are no longer valid
        // after this call.
        message EndSearchSolutionsRequest {
            string search_id = 1;
        }

        message EndSearchSolutionsResponse {}
        """
        search_id = request.search_id

        session = self.DB['search_sessions'].pop(search_id, dict())

        if session:
            searcher = session['searcher']
            searcher.stop()

            # cleanup pipelines
            for solution in searcher.solutions:
                self.DB['solutions'].pop(solution['id'], None)

            # while not searcher.done:
            #     time.sleep(1)

        return core_pb2.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        LOGGER.info("\n######## StopSearchSolutions ########\n%s########", request)
        """
        rpc StopSearchSolutions (StopSearchSolutionsRequest) returns (StopSearchSolutionsResponse) {}

        // Stops the search but leaves all currently found solutions available.
        // If the call is made in parallel with a running search and results are being streamed,
        // the "GetSearchSolutionsResults" stream is closed by the TA2 (as happens when the search
        // is concluded on its own). Search cannot be re-started after it has been stopped.
        message StopSearchSolutionsRequest {
            string search_id = 1;
        }

        message StopSearchSolutionsResponse {}
        """
        search_id = request.search_id

        session = self.DB['search_sessions'].get(search_id)

        if session:
            searcher = session['searcher']
            searcher.stop()

            # while not searcher.done:
            #     time.sleep(1)

        return core_pb2.StopSearchSolutionsResponse()

    def DescribeSolution(self, request, context):
        LOGGER.info("\n######## DescribeSolution ########\n%s########", request)
        """
        rpc DescribeSolution (DescribeSolutionRequest) returns (DescribeSolutionResponse) {}

        // Request a detailed description of the found solution.
        message DescribeSolutionRequest {
            string solution_id = 1;
        }

        message DescribeSolutionResponse {
            // A pipeline description. Nested pipelines should be fully described as well.
            PipelineDescription pipeline = 1;
            // Each step in a pipeline has description. These are reported in the order of steps in
            // the pipeline.
            repeated StepDescription steps = 2;
        }
        """
        solution_id = request.solution_id

        pipeline, session = self._get_pipeline(solution_id)

        description = pipeline.to_json_structure()
        # This is not working for some reason, so we make it "by hand"
        # response = encode_pipeline_description(pipeline, session['allowed_value_types'], '/tmp')
        # return response

        created_at = Timestamp()
        created_at.FromJsonString(description['created'])

        return core_pb2.DescribeSolutionResponse(
            pipeline=pipeline_pb2.PipelineDescription(
                id=description['id'],
                created=created_at,
                context=pipeline_pb2.PipelineContext.Value('PRODUCTION'),
                inputs=[
                    pipeline_pb2.PipelineDescriptionInput(
                        name=pipeline_input['name']
                    )
                    for pipeline_input in description['inputs']
                ],
                outputs=[
                    pipeline_pb2.PipelineDescriptionOutput(
                        name=output['name'],
                        data=output['data']
                    )
                    for output in description['outputs']
                ],
                steps=[
                    pipeline_pb2.PipelineDescriptionStep(
                        primitive=pipeline_pb2.PrimitivePipelineDescriptionStep(
                            primitive=primitive_pb2.Primitive(
                                id=step['primitive']['id'],
                                version=step['primitive']['version'],
                                python_path=step['primitive']['python_path'],
                                name=step['primitive']['name'],
                                digest=step['primitive']['digest']
                            ),
                            arguments={
                                key: pipeline_pb2.PrimitiveStepArgument(
                                    container=pipeline_pb2.ContainerArgument(
                                        data=value['data']
                                    )
                                )
                                for key, value in step['arguments'].items()
                            },
                            outputs=[
                                pipeline_pb2.StepOutput(
                                    id=output['id']
                                )
                                for output in step['outputs']
                            ],
                            hyperparams={
                                key: pipeline_pb2.PrimitiveStepHyperparameter(
                                    data=pipeline_pb2.DataArgument(
                                        data=str(value['data'])
                                    )
                                )
                                for key, value in step.get('hyperparams', dict()).items()
                            },
                        )
                    )
                    for step in description['steps']
                ]
            )
        )

    def _get_cv_args(self, configuration):
        method = core_pb2.EvaluationMethod.Name(configuration.method)
        if method == 'K_FOLD':
            return {
                'folds': configuration.folds,
                'stratified': configuration.stratified,
                'shuffle': configuration.shuffle,
                'random_seed': configuration.random_seed,
            }

        # Unsupported for now
        # elif method == 'HOLDOUT':
        #     train_test_ratio = configuration.train_test_ratio or 3
        #     test_size = 1 / (1 + train_test_ratio)
        #     cv = ShuffleSplit(
        #         n_splits=1,
        #         test_size=test_size,
        #         random_state=configuration.random_seed
        #     )

        else:
            raise ValueError(method)

    def _score_solution(self, searcher, dataset, problem, metric_pipelines, configuration):
        cv_args = self._get_cv_args(configuration)

        for metric, pipeline in metric_pipelines:
            metric = decode_performance_metric(metric)

            searcher.score_pipeline(dataset, problem, pipeline, [metric], **cv_args)

    def ScoreSolution(self, request, context):
        LOGGER.info("\n######## ScoreSolution ########\n%s########", request)
        """
        rpc ScoreSolution (ScoreSolutionRequest) returns (ScoreSolutionResponse) {}

        // Request solution to be scored given inputs. Inputs have to be Dataset container values
        // and pipeline outputs have to be predictions. It can internally run multiple fit + produce
        // runs of the pipeline on permutations of inputs data (e.g., for cross-validation). This is
        // also why we cannot expose outputs here.
        message ScoreSolutionRequest {
            string solution_id = 1;
            repeated Value inputs = 2;
            repeated ProblemPerformanceMetric performance_metrics = 3;
            // Any users associated with this call itself. Optional.
            repeated SolutionRunUser users = 4;
            ScoringConfiguration configuration = 5;
        }

        message ScoreSolutionResponse {
            string request_id = 1;
        }
        """
        solution_id = request.solution_id
        inputs = request.inputs
        performance_metrics = request.performance_metrics
        configuration = request.configuration

        # Still Ignored
        # users = request.users

        metric_pipelines = []
        for metric in performance_metrics:
            pipeline, _ = self._get_pipeline(solution_id)
            metric_pipelines.append((metric, pipeline))

        pipeline, session = self._get_pipeline(solution_id)

        dataset = Dataset.load(inputs[0].dataset_uri)
        problem = session['problem']
        searcher = session['searcher']
        allowed_value_types = session['allowed_value_types']

        self._start_session(
            pipeline.id,
            'score',
            self._score_solution,
            searcher,
            dataset,
            problem,
            metric_pipelines,
            configuration,
            metric_pipelines=metric_pipelines,
            problem=problem,
            allowed_value_types=allowed_value_types,
            configuration=configuration
        )

        return core_pb2.ScoreSolutionResponse(
            request_id=pipeline.id
        )

    def _get_score_solution_results(self, session, returned):
        problem = session['problem']
        allowed_value_types = session['allowed_value_types']
        configuration = session['configuration']
        targets = problem['inputs'][0]['targets']
        dataset_id = problem['inputs'][0]['dataset_id']

        metric_pipelines = session['metric_pipelines']
        metric_fold_scores = []
        for metric, pipeline in metric_pipelines:
            for fold, score in enumerate(pipeline.cv_scores):
                metric_fold_scores.append({
                    'metric': problem['problem']['performance_metrics'][0],
                    'fold': fold,
                    'value': score,
                    'targets': targets,
                    'dataset_id': dataset_id,
                    'random_seed': configuration.random_seed
                })

        if len(metric_fold_scores) > returned:
            return core_pb2.GetScoreSolutionResultsResponse(
                progress=self._get_progress(session),
                scores=[
                    encode_score(score, allowed_value_types, '/tmp')
                    for score in metric_fold_scores
                ],
            )

    def GetScoreSolutionResults(self, request, context):
        LOGGER.info("\n######## GetScoreSolutionResults ########\n%s########", request)
        """
        rpc GetScoreSolutionResults (GetScoreSolutionResultsRequest) returns (stream GetScoreSolutionResultsResponse) {}

        // Get all score results computed until now and start receiving any
        // new score results computed as well.
        message GetScoreSolutionResultsRequest {
            string request_id = 1;
        }

        message GetScoreSolutionResultsResponse {
            // Overall process progress.
            Progress progress = 1;
            // List of score results. List can be incomplete while the process is in progress.
            repeated Score scores = 2;
        }
        """
        request_id = request.request_id

        session = self.DB['score_sessions'][request_id]
        if not session:
            raise ValueError('Invalid request_id')

        return self._stream(session, self._get_score_solution_results, close_on_done=True)

    def _get_pipeline(self, solution_id, fitted=False):
        prefix = 'fitted_' if fitted else ''
        solution = self.DB[prefix + 'solutions'].get(solution_id)
        if not solution:
            raise ValueError('Invalid {}solution_id'.format(prefix))

        if fitted:
            return solution

        else:
            session = solution.pop('session')
            pipeline = Pipeline.from_json_structure(solution)
            pipeline.cv_scores = list()
            pipeline.score = solution.get('score')
            pipeline.normalized_score = solution.get('normalized_score')
            solution['session'] = session

            return pipeline, session

    def _fit_solution(self, pipeline, dataset, problem, exposed_outputs):

        runtime = Runtime(
            pipeline=pipeline,
            problem_description=problem,
            context=Context.TESTING,
            volumes_dir=self.static,
        )

        fit_results = runtime.fit(inputs=[dataset])
        fit_results.check_success()

        for exposed_name, exposed_details in exposed_outputs.items():
            csv_path = urlparse(exposed_details.csv_uri).path
            fit_results.values[exposed_name].to_csv(csv_path, index=None)

        self.DB['fitted_solutions'][pipeline.id] = runtime

    def FitSolution(self, request, context):
        LOGGER.info("\n######## FitSolution ########\n%s########", request)
        """
        rpc FitSolution (FitSolutionRequest) returns (FitSolutionResponse) {}

        // Fit the solution on given inputs. If a solution is already fitted on inputs this is a NOOP
        // (if no additional outputs should be exposed). This can happen when a TA2 simultaneously
        // fits the solution as part of the solution search phase.
        message FitSolutionRequest {
            string solution_id = 1;
            repeated Value inputs = 2;
            // List of data references of step outputs which should be exposed to the TA3 system.
            // If you want to expose outputs of the whole pipeline (e.g., predictions themselves),
            // list them here as well. These can be recursive data references like
            // "steps.1.steps.4.produce" to point to an output inside a sub-pipeline.
            // Systems only have to support exposing final outputs and can return "ValueError" for
            // intermediate values.
            repeated string expose_outputs = 3;
            // Which value types should be used for exposing outputs. If not provided, the allowed
            // value types list from hello call is used instead.
            // The order is important as TA2 system will try value types in order until one works out,
            // or an error will be returned instead of the value. An error exposing a value does not
            // stop the overall process.
            repeated ValueType expose_value_types = 4;
            // Any users associated with this call itself. Optional.
            repeated SolutionRunUser users = 5;
        }

        message FitSolutionResponse {
            string request_id = 1;
        }
        """
        solution_id = request.solution_id
        inputs = request.inputs

        # Still Ignored
        # users = request.users

        pipeline, session = self._get_pipeline(solution_id)

        dataset = Dataset.load(inputs[0].dataset_uri)
        exposed_outputs = self._get_exposed_outputs(pipeline, request)

        problem = session['problem']

        self._start_session(
            pipeline.id,
            'fit',
            self._fit_solution,
            pipeline,
            dataset,
            problem,
            exposed_outputs,
            pipeline=pipeline,
            exposed_outputs=exposed_outputs
        )

        return core_pb2.FitSolutionResponse(
            request_id=pipeline.id
        )

    def _get_fit_solution_results(self, session, returned):
        """Get fitted solution only if completed or it's the first try.

        Later on we might want to giva response for each done step.
        """
        done = session.get('done')
        if done or not returned:
            exposed_outputs = session['exposed_outputs'] if done else None
            fitted_solution_id = session['id'] if done else None

            return core_pb2.GetFitSolutionResultsResponse(
                progress=self._get_progress(session),
                steps=[
                    # In the future there will be one of these for each primitive
                    # core_pb2.StepProgress(
                    #     progress=core_pb2.Progress(
                    #         state=core_pb2.ProgressState.Value('COMPLETED'),
                    #         status=error,
                    #         start=None,
                    #         end=None
                    #     )
                    # )
                ],
                exposed_outputs=exposed_outputs,
                fitted_solution_id=fitted_solution_id
            )

    def GetFitSolutionResults(self, request, context):
        LOGGER.info("\n######## GetFitSolutionResults ########\n%s########", request)
        """
        rpc GetFitSolutionResults (GetFitSolutionResultsRequest) returns (stream GetFitSolutionResultsResponse) {}

        // Get all fitted results currently available and start receiving any further
        // new fitted results as well.
        message GetFitSolutionResultsRequest {
            string request_id = 1;
        }

        message GetFitSolutionResultsResponse {
            // Overall process progress.
            Progress progress = 1;
            // The list contains progress for each step in the pipeline, in order.
            // List can be incomplete while the process is in progress. Systems can provide
            // steps only at the end (when "progress" equals COMPLETED) and not during running.
            repeated StepProgress steps = 2;
            // A mapping between data references of step outputs and values.
            map<string, Value> exposed_outputs = 3;
            // The fitted solution ID, once progress = COMPLETED.
            string fitted_solution_id = 4;
        }
        """
        request_id = request.request_id

        session = self.DB['fit_sessions'][request_id]
        if not session:
            raise ValueError('Invalid request_id')

        return self._stream(session, self._get_fit_solution_results, close_on_done=True)

    def _get_exposed_outputs(self, pipeline, request, produce_id=None, default=False):
        expose_outputs = request.expose_outputs
        expose_value_types = request.expose_value_types

        if not expose_outputs and not default:
            # Skip the rest of steps
            return None

        steps = len(pipeline.steps)
        last_step_produce = 'steps.{}.produce'.format(steps - 1)
        output_0 = 'outputs.0'

        if expose_outputs:
            if expose_outputs not in [[output_0], [last_step_produce]]:
                raise ValueError("Exposing partial outputs not supported.'")

            exposed_output = expose_outputs[0]

        else:
            exposed_output = output_0

        if expose_value_types:
            csv_uri = value_pb2.ValueType.Value('CSV_URI')
            if csv_uri not in expose_value_types:
                raise ValueError(
                    "Unsupported ValueTypes: {}".format(expose_value_types)
                )
            elif len(expose_value_types) > 1:
                LOGGER.warn("Unsupported ValueTypes: %s",
                            [evt for evt in expose_value_types if evt != csv_uri])

        produce_id = produce_id or pipeline.id

        predictions_path = os.path.join(self.output_dir, 'predictions', pipeline.id)
        os.makedirs(predictions_path, exist_ok=True)

        csv_name = '{}.{}.csv'.format(produce_id, exposed_output)
        csv_path = os.path.join(predictions_path, csv_name)
        csv_uri = 'file://' + csv_path

        exposed_outputs = {
            exposed_output: value_pb2.Value(csv_uri=csv_uri)
        }

        return exposed_outputs

    def _get_solution(self, solution_id):
        solution = self.DB['solutions'].get(solution_id)
        if not solution:
            raise ValueError('Invalid solution_id')

        return solution

    def _get_fitted_solution(self, solution_id):
        fitted_solutions = self.DB['fitted_solutions']
        runtime = fitted_solutions.get(solution_id)
        if not runtime:
            LOGGER.error(list(fitted_solutions.keys()))
            raise ValueError('Invalid fitted_solution_id')

        return runtime

    def _produce_solution(self, runtime, dataset, exposed_outputs):
        produce_results = runtime.produce(inputs=[dataset])
        produce_results.check_success()

        for exposed_name, exposed_details in exposed_outputs.items():
            csv_path = urlparse(exposed_details.csv_uri).path
            produce_results.values[exposed_name].to_csv(csv_path, index=None)

    def ProduceSolution(self, request, context):
        LOGGER.info("\n######## ProduceSolution ########\n%s########", request)
        """
        rpc ProduceSolution (ProduceSolutionRequest) returns (ProduceSolutionResponse) {}

        // Produce (execute) the solution on given inputs. A solution has to have been fitted for this
        // to be possible (even if in cases where this is just created by transformations).
        message ProduceSolutionRequest {
            string fitted_solution_id = 1;
            repeated Value inputs = 2;
            // List of data references of step outputs which should be exposed to the TA3 system.
            // If you want to expose outputs of the whole pipeline (e.g., predictions themselves),
            // list them here as well. These can be recursive data references like
            // "steps.1.steps.4.produce" to point to an output inside a sub-pipeline.
            // Systems only have to support exposing final outputs and can return "ValueError" for
            // intermediate values.
            repeated string expose_outputs = 3;
            // Which value types should be used for exposing outputs. If not provided, the allowed
            // value types list from a hello call is used instead.
            // The order is important as the TA2 system will try value types in order until one works
            // out, or an error will be returned instead of the value. An error exposing a value does
            // not stop the overall process.
            repeated ValueType expose_value_types = 4;
            // Any users associated with this call itself. Optional.
            repeated SolutionRunUser users = 5;
        }

        message ProduceSolutionResponse {
            string request_id = 1;
        }
        """
        fitted_solution_id = request.fitted_solution_id
        inputs = request.inputs

        # Still Ignored
        # users = request.users

        runtime = self._get_fitted_solution(fitted_solution_id)
        dataset = Dataset.load(inputs[0].dataset_uri)

        produce_id = str(uuid.uuid4())
        exposed_outputs = self._get_exposed_outputs(runtime.pipeline, request, produce_id, True)

        self._start_session(
            produce_id,
            'produce',
            self._produce_solution,
            runtime,
            dataset,
            exposed_outputs,
            runtime=runtime,
            exposed_outputs=exposed_outputs
        )

        return core_pb2.ProduceSolutionResponse(
            request_id=produce_id
        )

    def _get_produce_solution_results(self, session, returned):
        done = session.get('done')
        if done or not returned:
            exposed_outputs = session['exposed_outputs'] if done else None

            return core_pb2.GetProduceSolutionResultsResponse(
                progress=self._get_progress(session),
                steps=[
                    # In the future there will be one of these for each primitive
                    # core_pb2.StepProgress(
                    #     progress=core_pb2.Progress(
                    #         state=core_pb2.ProgressState.Value('COMPLETED'),
                    #         status=error,
                    #         start=None,
                    #         end=None
                    #     )
                    # )
                ],
                exposed_outputs=exposed_outputs
            )

    def GetProduceSolutionResults(self, request, context):
        LOGGER.info("\n######## GetProduceSolutionResults ########\n%s########", request)
        """
        rpc GetProduceSolutionResults (GetProduceSolutionResultsRequest) returns (stream GetProduceSolutionResultsResponse) {}

        // Get all producing results computed until now and start receiving any
        // new producing results computed as well.
        message GetProduceSolutionResultsRequest {
            string request_id = 1;
        }

        message GetProduceSolutionResultsResponse {
            // Overall process progress.
            Progress progress = 1;
            // The list contains progress for each step in the pipeline, in order.
            // List can be incomplete while the process is in progress. Systems can provide
            // steps only at the end (when "progress" equals COMPLETED) and not during running.
            repeated StepProgress steps = 2;
            // A mapping between data references of step outputs and values.
            map<string, Value> exposed_outputs = 3;
        }
        """
        request_id = request.request_id

        session = self.DB['produce_sessions'][request_id]
        if not session:
            raise ValueError('Invalid request_id')

        return self._stream(session, self._get_produce_solution_results, close_on_done=True)

    def SolutionExport(self, request, context):
        LOGGER.info("\n######## SolutionExport ########\n%s########", request)
        """
        rpc SolutionExport (SolutionExportRequest) returns (SolutionExportResponse) {}

        // Exports a solution for evaluation purposes based on NIST specifications.
        message SolutionExportRequest {
            // Found solution to export.
            DEPRECTAED: string fitted_solution_id = 1;
            string solution_id = 1;    # From 2019.2.27
            // Solution rank to be used for the exported solution. Lower numbers represent
            // better solutions. Presently NIST requirements are that ranks should be non-negative
            // and that each exported pipeline have a different rank. TA3 should make sure not to repeat ranks.
            // Filenames of exported files are left to be chosen by the TA2 system.
            double rank = 2;
        }

        message SolutionExportResponse {}
        """
        solution_id = request.solution_id
        rank = request.rank

        solution = self._get_solution(solution_id)

        dump_pipeline(solution, self.ranked_dir, rank=rank)

        return core_pb2.SolutionExportResponse()

    def UpdateProblem(self, request, context):
        LOGGER.info("\n######## UpdateProblem ########\n%s########", request)
        """
        rpc UpdateProblem (UpdateProblemRequest) returns (UpdateProblemResponse) {}

        // Updates problem with new description. This also updates the problem description for all
        // ongoing solution searches associated with this problem. Internal behavior of TA2
        // is unspecified: it can simply start a new search using new problem description, or
        // it can start modifying solutions it has already found to new problem description, or
        // it can use it to further help narrow down ongoing solution searches. In any case, after
        // this call returns, all reported solutions for searches associated with this problem
        // should be for the updated problem description.
        message UpdateProblemRequest {
            string search_id = 1;
            // New problem description. It has to be provided in full and it replaces existing
            // problem description.
            ProblemDescription problem = 2;
        }

        message UpdateProblemResponse {}
        """
        # Ignored
        # search_id = request.search_id
        # problem = request.problem

        return core_pb2.UpdateProblemResponse()

    def ListPrimitives(self, request, context):
        LOGGER.info("\n######## ListPrimitives ########\n%s########", request)
        """
        rpc ListPrimitives (ListPrimitivesRequest) returns (ListPrimitivesResponse) {}

        // List all primitives known to TA2, their IDs, versions, names, and digests. Using this
        // information a TA3 should know which primitives may be put into a pipeline template.
        // To narrow down potential primitives to use a TA3 can also ask a TA2 to do a solution
        // search and then observe which primitives the TA2 is using. If more metadata about primitives
        // is needed, then a TA3 can use the results of this call to map primitives to metadata
        // (from Python code or primitive annotations) on its own.
        message ListPrimitivesRequest {}

        message ListPrimitivesResponse {
            repeated Primitive primitives = 1;
        }

        // Description of the primitive.
        message Primitive {
            string id = 1;
            string version = 2;
            string python_path = 3;
            string name = 4;
            // Digest is optional, because some locally registered primitives might not have it.
            // But for all primitives published it is available and it should be provided here as well.
            string digest = 5;
        }
        """
        primitive = primitive_pb2.Primitive(
            id="dummy",
            version="dummy",
            python_path="dummy",
            name="dummy",
            digest="dummy"
        )
        return core_pb2.ListPrimitivesResponse(
            primitives=[primitive]
        )

    def Hello(self, request, context):
        LOGGER.info("\n######## Hello ########\n%s########", request)
        """
        rpc Hello (HelloRequest) returns (HelloResponse) {}

        message HelloResponse {
        // Some string identifying the name and version of the TA2 system.
        string user_agent = 1;
        // Shall be set to "protocol_version" above.
        string version = 2;
        // List of value types that a TA3 system can use to communicate values to a TA2 system.
        // The order is important as a TA3 system should try value types in order until one works
        // out, or an error will be returned instead of the value.
        repeated ValueType allowed_value_types = 3;
        // List of API extensions that a TA2 supports.
        repeated string supported_extensions = 4;
        }
        """

        return core_pb2.HelloResponse(
            user_agent='MIT-FL-TA2',
            version=VERSION,
            allowed_value_types=[
                value_pb2.ValueType.Value('DATASET_URI'),
                value_pb2.ValueType.Value('CSV_URI')
            ]
        )
