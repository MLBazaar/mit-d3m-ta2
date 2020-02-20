<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![Release Shield](https://img.shields.io/github/release-pre/HDI-Project/mit-d3m-ta2.svg)](https://github.com/HDI-Project/mit-d3m-ta2/releases)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/mit-d3m-ta2.svg?branch=master)](https://travis-ci.org/HDI-Project/mit-d3m-ta2)


# MIT-D3M-TA2

MIT-Featuretools TA2 submission for the D3M program.

- Free software: MIT license
- Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
- Documentation: https://HDI-Project.github.io/mit-d3m-ta2

# Overview

This repository contains the TA2 submission for the [Data Driven Discovery of Models (D3M) DARPA
program](https://www.darpa.mil/program/data-driven-discovery-of-models) developed by the DAI-Lab
and Featuretools teams.

# Install

## Requirements

**mit-d3m-ta2** has been developed and tested on [Python 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **mit-d3m-ta2** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **mit-d3m-ta2**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) mit-d3m-ta2-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source mit-d3m-ta2-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **mit-d3m-ta2**!

## Install the latest release

In order to install **mit-d3m-ta2**, you will have to clone the repository
and checkout its stable branch:

```bash
git clone git@github.com:HDI-Project/mit-d3m-ta2.git
cd mit-d3m-ta2
git checkout stable
```

Once done, make sure to having created and activated your virtalenv and then simply execute:

```
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

First, please head to [the GitHub page of the project](https://github.com/HDI-Project/mit-d3m-ta2)
and make a fork of the project under you own username by clicking on the **fork** button on the
upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes
the number of the issue that you are going to work on:

```bash
git clone git@github.com:{your username}/mit-d3m-ta2.git
cd mit-d3m-ta2
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, which will install some additional
dependencies for code linting and testing.

```bash
make install-develop
```

Make sure to use them regularly while developing by running the commands `make lint` and `make test`.

### Additional Dependencies

Additional dependencies required to execute some of the TA1 primitives have been left out from
the command above in order to keep maximum compatibility with the different types of systems
and avoid dependency conflicts.

Because of this, some datasets, including timeseries and image data modalities, might not work
properly.

In order to make them work, install the additional dependencies and download additional files
with the following commands:

```bash
sudo apt-get install $(cat system_requirements.txt)
pip install -r devel_requirements.txt
mkdir -p static
python -m d3m.index download -o static
```

And keep in mind the following considerations:

* The command line script `ta2` explained in the usage section below will stop working and will
  need to be replaced with `python -m ta2` in all the examples.
* Some red warnings might show in the command line indicating that incompatible versions have been
  install. These warnings can be safely ignored, as their only consequence is the previous point.


# Data Format

**mit-d3m-ta2** runs on datasets in the [D3M Format](https://github.com/mitll/d3m-schema)

## Datasets Collection

You can find a collection of datasets in the D3M format in the [d3m-data-dai S3 Bucket in
AWS](https://d3m-data-dai.s3.amazonaws.com/index.html), including the corresponding `TRAIN`,
`TEST` and `SCORE` partitions following the schema specification.

More datasets in newer versions of the schema can also be found in [the private datasets
repository](https://gitlab.datadrivendiscovery.org/d3m/datasets).

## D3M Seed Datasets

Our TA2 system is regularly evaluated over the collection of [Seed Datasets found in the private
datasets repostory](https://gitlab.datadrivendiscovery.org/d3m/datasets/tree/master/seed_datasets_current).

As specified in the `README` file form this repository, you will need [git-lfs](https://git-lfs.github.com/)
in order to download all the included files.

Note that the complete collection of seed datasets is around 60 GB big, so the recommended approach
is to download only those parts of the repository that will be used following the instructions in
the [Partial Downloading section](https://gitlab.datadrivendiscovery.org/d3m/datasets#partial-downloading)

Once downloaded, the local testing commands can be used passing the `seed_datasets_current` root
folder path to the `--input` option.

Example: `--input /path/to/d3m/datasets/repo/seed_datasets_current`

# Leaderboard

The following leaderboard has been built using the `TA2 Standalone Mode` with `2` as
the maximum number of tuning iterations to perform (`budget`) and `30` as the maximum time
allowed for the tuning (`timeout`):

| dataset                        | template                                    |   cv_score |   test_score |   elapsed_time |   tuning_iterations | data_modality   | task_type      |
|--------------------------------|---------------------------------------------|------------|--------------|----------------|---------------------|-----------------|----------------|
| 30_personae                    | gradient_boosting_classification.all_hp.yml | 0.728894   |     0.619048 |        5.93087 |                   2 | single_table    | classification |
| 57_hypothyroid                 | gradient_boosting_classification.all_hp.yml | 0.862681   |     0.981003 |       38.6418  |                   2 | single_table    | classification |
| 185_baseball                   | gradient_boosting_classification.all_hp.yml | 0.646959   |     0.675132 |       17.3313  |                   2 | single_table    | classification |
| 313_spectrometer               | gradient_boosting_classification.all_hp.yml | 0.281409   |     0.304201 |       45.3676  |                   2 | single_table    | classification |
| 27_wordLevels                  | gradient_boosting_classification.all_hp.yml | 0.268882   |     0.288937 |      169.197   |                   2 | single_table    | classification |
| 1491_one_hundred_plants_margin | gradient_boosting_classification.all_hp.yml | 0.00957403 |     0.451364 |      114.561   |                   2 | single_table    | classification |

This table can be also downloaded as a [CSV file](leaderboard.csv)

# Usage

## Local Testing

Two scripts are included in the repository for local testing:

### TA2 Standalone Mode

The TA2 Standalone mode can be executed locally using the `ta2` command line interface.

To use this, run the `ta2 test` command passing one or more dataset names as positional
arguments as well as either a budget. `-b`, or a timeout, `-t`.

For example, in order to process the datasets `185_baseball` and `196_autoMpg` during 60 seconds
each, the following command would be used:

```
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

### TA2-TA3 Server Mode

The TA2-TA3 API mode can be executed using the `ta2 server` command, as well as any of the
optional named arguments required.

This will start a ta2 server in the background ready to serve requests from a ta3 client.

```
ta2 server
```

For a full description of the script options, execute `ta2 server --help`.

### TA2-TA3 Test

In order to test the TA2-TA3 Server, a convenience `ta3` command line interface has been included,
which allows testing one or more datasets by issuing a predefined sequence of calls to the
TA2-TA3 Server.

To use it, run the `ta2 ta3` command passing one or more dataset names as positional
arguments, as well as any of the optional arguments.

For example, in order to process the datasets `185_baseball` and `196_autoMpg` during 60 seconds
each, the following command would be used:

```
ta2 ta3 -t60 185_baseball 196_autoMpg
```

**NOTE**: In order to be able to execute this command, a `ta2 server` process must be already
running in the same machine.

This will start sending requests to the `ta3-server` to search and tune the best pipeline
possible for each dataset during a maximum of 60 seconds.

For a full description of the script options, execute `ta2 ta3 --help`.

Also remember that a TA2-TA3 Server must be running when you execute this script!

## Docker Usage

In order to run TA2-TA3 server from docker, you first have to build the image and
execute the `run_docker.sh` script.
After that, in a different console, you can run the `ta3` script passing it the
`--docker` flag to adapt the input paths accordingly:

```
make build
./run_docker.sh
```

And, in a different terminal:

```
ta2 ta3 --docker <OPTIONS>
```

# Submission

The submission steps are defined here: https://datadrivendiscovery.org/wiki/display/gov/Submission+Procedure+for+TA2

In our case, the submission steps consist of:

1. Execute the `make submit` command locally. This will build the docker image and push it to the
   gitlab registry.
2. Copy the `kubernetes/ta2.yaml` file to the Jump Server and execute the validation command `/performer-toolbox/d3m_runner/d3m_runner.py --yaml-file ta2.yaml --mode ta2 --debug`
3. If successful, copy the `ta2.yaml` file over to the submission repository folder and commit/push it.

For winter-2019 evaluation, the submission repository was https://gitlab.datadrivendiscovery.org/ta2-submissions/ta2-mit/may2019
