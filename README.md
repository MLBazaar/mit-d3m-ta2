# MIT TA2

MIT-Featuretools TA2 submission for the D3M program.

## Setup

To install the project simply execute the following command:

```
make install
```

For development, use the `make install-develop` command instead, which will install the project
in editable mode and also install some additional code linting tools.

## Local Testing

Two scripts are included in the repository for local testing:

### TA2 Standalone Mode

The TA2 Standalone mode can be executed locally using the `ta2_test.py` script.

To use this script, call it using python and passing one or more dataset names
as positional arguments, along with any of the optional named arguments.

```
python ta2_test.py -b10 -t60 -v 185_baseball 196_autoMpg
```

For a full description of the script options, execute `python ta2_test.py --help`.

### TA2-TA3 API Mode

The TA2-TA3 API mode can be executed locally using the `ta3_test.py` script.

This script will start a ta2 server in the background and then send a series of requests
using the ta3 client to fully test a dataset.

To use this script, call it using python and passing one or more dataset names
as positional arguments, along with any of the optional named arguments.

```
python ta3_test.py -v 185_baseball 196_autoMpg
```

By default, the logs of the server will be stored inside the `logs` folder, and the output
from the client will be shown in stdout, but this behavior can be optionally changed by
passing additional arguments.

Optionally, the server can be prevented from being started in the background by using the
`--no-server` flag. This is useful if you are running the server in a separated process.

For a full description of the script options, execute `python ta3_test.py --help`.
