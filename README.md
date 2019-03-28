# MIT TA2

MIT-Featuretools TA2 submission for the D3M program.

## Setup

To install the project simply execute the following command:

```
make install
```

For development, use the `make install-develop` command instead, which will install the project
in editable mode and also install some additional code linting tools.

## Datasets

For development and evaluation of pipelines, we use two kind of datasets:
* [D3M seed datasets](https://gitlab.datadrivendiscovery.org/d3m/datasets/): A growing set of seed datasets
released to D3M performers. These datasets have been converted to D3M format and schematized.
* [D3M data dai datasets](https://d3m-data-dai.s3.amazonaws.com/index.html): A custom formatted version
by [DAI Group](https://dai.lids.mit.edu/) of training
datasets provided in [D3M datasets](https://gitlab.datadrivendiscovery.org/d3m/datasets/) to help D3M performers
develop approaches to automatically create machine learning solutions to a variety of problems. Compared to the
original datasets version, these datasets are already split in `SCORE`, `TEST` and `TRAIN` directories using the
same approach as the seed datasets.

### D3M Seed Datasets

Before start, make sure you have the proper rights on the D3M datasets
[repo](https://gitlab.datadrivendiscovery.org/d3m/datasets). You will need credentials and read permissions
to download them.

As specified in the `README` file, you will need [git-lfs](https://git-lfs.github.com/) to download files faster.
As all datasets are around 54 GB, the recommended approach is to download only parts of the repository as needed, following
instructions in the [Partial Downloading](https://gitlab.datadrivendiscovery.org/d3m/datasets#partial-downloading)
section.

Once downloaded the specific datasets, the local testing commands can be used with the `--input` option and the correspoding
path. Example: `--input /path/to/d3m/datasets/repo/seed_datasets_current`

#### Leaderboard

The following leaderboard has been built using the `TA2 Standalone Mode` with `10` as
the maximum number of tuning iterations to perform (`budget`) and `60s` as the maximum time
allowed for the tuning (`timeout`).

| dataset | CV score | rank |
|------|-------|------|
| 299_libras_move | 27.740541 | 1 |
| 196_autoMpg | 19.369635 | 2 |
| 30_personae | 0.728893 | 3 |
| 185_baseball | 0.646959 | 4 |
| 27_wordLevels | 0.258386 | 5 |
| 1567_poker_hand | 0.179153 | 6 |
| 26_radon_seed | 0.030257 | 7 |
| 1491_one_hundred_plants_margin | 0.009574 | 8 |
| 1491_one_hundred_plants_margin_clust | N/S | N/S |
| 22_handgeometry | N/A | N/A |

## Local Testing

Two scripts are included in the repository for local testing:

### TA2 Standalone Mode

The TA2 Standalone mode can be executed locally using the `ta2_test.py` script.

To use this script, call it using python and passing one or more dataset names
as positional arguments, along with any of the optional named arguments.

```
python ta2_test.py -b10 -t60 -v 185_baseball
```

For a full description of the script options, execute `python ta2_test.py --help`.

### TA2-TA3 API Mode

The TA2-TA3 API mode can be executed locally using the `ta3_test.py` script.

This script will start a ta2 server in the background and then send a series of requests
using the ta3 client to fully test a dataset.

To use this script, call it using python and passing one or more dataset names
as positional arguments, along with any of the optional named arguments. If no dataset
names are given, all the datasets found in the input folder will be tested in succession.

```
python ta3_test.py -v 185_baseball
```

By default, the logs of the server will be stored inside the `logs` folder, and the output
from the client will be shown in stdout, but this behavior can be optionally changed by
passing additional arguments.

Optionally, the server can be prevented from being started in the background by using the
`--no-server` flag. This is useful if you are running the server in a separated process.

For a full description of the script options, execute `python ta3_test.py --help`.

### Docker run

In order to run TA2-TA3 server from docker, you first have to build the image and
execute the `run_docker.sh` script.
After that, in a different console, you can run the `ta3_test.py` script passing it the
`--docker` flag to adapt the input paths accordingly:

```
make build
./run_docker.sh
```

And, in a different terminal:

```
python ta3_test.py -v -t2 --docker
```

## Submission

The submission steps are defined here: https://datadrivendiscovery.org/wiki/display/gov/Submission+Procedure+for+TA2

In our case, the submission steps consist of:

1. Execute the `make submit` command locally. This will build the docker image and push it to the
   gitlab registry.
2. Copy the `kubernetes/ta2.yaml` file to the Jump Server and execute the validation command `/performer-toolbox/d3m_runner/d3m_runner.py --yaml-file ta2.yaml --mode ta2 --debug`
3. If successful, copy the `ta2.yaml` file over to the submission repository folder and commit/push it.

For winter-2019 evaluation, the submission repository was https://gitlab.datadrivendiscovery.org/ta2-submissions/ta2-mit/winter-2019
