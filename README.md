<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![Release Shield](https://img.shields.io/github/release-pre/HDI-Project/mit-d3m-ta2.svg)](https://github.com/HDI-Project/mit-d3m-ta2/releases)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/mit-d3m-ta2.svg?branch=master)](https://travis-ci.org/HDI-Project/mit-d3m-ta2)


# MIT-D3M-TA2

TA2 submission for the D3M program built by the Data to AI Lab, MIT and Featurelabs team.

* Free software: [MIT license](https://github.com/HDI-Project/mit-d3m-ta2/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Homepage: https://github.com/HDI-Project/mit-d3m-ta2
* Documentation: https://HDI-Project.github.io/mit-d3m-ta2

## Overview

**mit-d3m-ta2** is an Auto Machine Learning platform prepared to run on datasets stored in the
[D3M Format](https://github.com/mitll/d3m-schema), wich can be used as a simple Python library,
as a standalone end-to-end command-line application or as a GRPC powered Backend platform for
client-server architectures.

**mit-d3m-ta2** is also the TA2 submission for the [Data Driven Discovery of Models (D3M) DARPA
program](https://www.darpa.mil/program/data-driven-discovery-of-models) developed by the DAI-Lab
and Featuretools teams.

# Install

## System Requirements

* [Python 3.6](https://www.python.org/downloads/).
* [Docker](https://docs.docker.com/install/).

## Install with pip

The easiest and recommended way to install **mit-dem-ta2** is using [pip](
https://pip.pypa.io/en/stable/):

```bash
pip install mit-d3m-ta2
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://hdi-project.github.io/MLBlocks/contributing.html#get-started).

# Usage

Upon installing **mit-d3m-ta2** a command line interface called `ta2` will be installed on
your environment.

This CLI several ejecution modes.

## Standalone Mode

The Standalone mode searches for the best possible pipeline for a given Machine Learning
problem and returns details about the best pipeline found and the scores obtained.

To run in this mode, execute the `ta2 standalone` command passing one or more dataset names as
positional arguments, as well as either a maximum number of tuning iterations to perform
(`-b`, `--budget`) or a maximum time to tune, in seconds (`-t`, `--timeout`).

In this example we process the datasets `185_baseball` and `196_autoMpg` during 60 seconds each:

```bash
ta2 test -t60 185_baseball 196_autoMpg
```

This will start searching and tuning the best pipeline possible for each dataset during a maximum
of 60 seconds and, at the end, print a table with all the results on stdout.

Additionally, the following options can be passed:

* `-i INPUT_PATH`: Path to the folder where the datasets can be found. Defaults to `input`.
* `-o OUTPUT_PATH`: Path to the folder where the output pipeliens will be saved. Defaults to `output`.
* `-b BUDGET`: Maximum number of tuning iterations to perform.
* `-t TIMEOUT`: Maximum allowed time for the tuning, in seconds.
* `-a, --all`: Process all the datasets found in the input folder.
* `-v, --verbose`: Set logs to INFO level. Use it twice to increase verbosity to DEBUG.
* `-r CSV_PATH`: Store the results in the indicated CSV file instead of printing them on stdout.
* `-s STATIC_PATH`: Path to a directory with static files required by primitives. Defaults to `static`.

For a full description of the options, execute `ta2 test --help`.

## Server Mode

The Server mode starts a GRPC server using the TA3TA2 API, ready to work as a backend for
a D3M TA3 User Interface.

To run in this mode, execute the command `ta2 server`.

```bash
ta2 server
```

For a full description of the script options, execute `ta2 server --help`.
