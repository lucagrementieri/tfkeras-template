# TF.Keras Template

Code and documentation template for deep learning projects based on tf.keras.
This repository is intended to be cloned at the beginning of any
new deep learning project based on Tensorflow 2 and Keras.

Every TODO comment in the code indicates a portion of the code
that should be adapted for every specific project.
The rest of the code should usually remain almost unchanged.

## Getting Started

These instructions will get you a copy of the project up and running
on your local machine for development and testing purposes.

The project folder, including also files excluded from git versioning,
has the following structure:

```
tfkeras-template/                   [main folder]
│   .gitignore                      [files ignored by git]
│   generate_data.py                [script to simulate data]
│   dataset_statistics.py           [script to compute dataset mean and standard deviation]
│   LICENSE                         [code license]
│   README.md                       [this file]
│   requirements.txt                [package dependencies]
│   setup.py                        [package setup script]
│
├───data                            [data folder excluded from git tracking]
│   │   targets.csv                 [targets for train, dev and test data]
│   │
│   ├───train
│   │       ...
│   ├───dev
│   │       ...
│   └───test
│           ...
│
└───tfkeras_template             [package source code folder]
        ...
```

You should comply to this structure in your project,
in particular you should structure the `data` folder containing your dataset
according to the hierarchy shown. 

### Prerequisites

In order to run the code you need to have Python 3.6+ installed.

### Installing

You can install the package on MacOS/Linux with the following commands:
```
git clone https://github.com/lucagrementieri/tfkeras-template.git
cd tfkeras-template
python3 setup.py sdist
python3 setup.py bdist_wheel
pip3 install --no-index --find-links=dist tfkeras_template -r requirements.txt
```
The installation is not compulsory, you can run all the commands
explained below by calling them inside the directory `tfkeras-template`.

Here data are synthetic so to generate them you have to run:
```
python3 generate_data.py
```
This script should be removed from your project.

You can compute the statistics (mean and standard deviation) of
the generated training dataset with the script `dataset_statistics.py`.
The calculated values are saved in `data/statistics.json` and they can be
used to replace the hard-coded values used for data normalization in `app.py`. 

## Usage

A command line interface is available to easily interact with the package.
It is defined in the file `__main__.py` file of the package.

This file allows to execute the package passing the flag `-m` to the Python interpreter
For example, we can invoke the help page of the package
```
python3 -m tfkeras_template --help
``` 

The help page lists the available commands:
- `ingest`: preprocess raw data and export it in a suitable format for model
training;
- `train`: train the deep learning model on ingested data;
- `eval`: evaluate the model on ingested validation data;
- `test`: produce model output on a single raw data sample.

Every command has its separate help page that can be visualized with
```
python3 -m tfkeras_template <command> --help
```

### Command `ingest`

The ingestion phase is useful if preprocessing is computationally expensive and
many transformations are required. Here, for example, it is not really necessary
but it is included to show the code structure.

In some cases an additional `safe-ingest` can be used to check and assure labels 
coherence among the different dataset splits (training/development/test data) or to perform transformations
that depend on other splits. Here it is not needed because the
set of labels is not fixed since the example task is a regression.

#### Examples

Only the training set and the development set have to be ingested
and that can be do with the following lines:
```
python3 -m tfkeras_template ingest data train
python3 -m tfkeras_template ingest data dev
```

For more details on the usage you can access the help page with the command
```
python3 -m tfkeras_template ingest --help
```

`ingest` command can be called on test data with
```
python3 -m tfkeras_template ingest data test
```
but it produces an empty directory because test data enters
the system as raw data.
Data processing on test data is performed at runtime
before the model prediction.

### Command `train`

The training phase has always the same structure and the template is built
to keep all the tried models in files separated from the main training function.

There are many ways to define a model in Tensorflow 2.0 with Keras.
The template supports both symbolic (traditional Tensorflow) and imperative
(Eager Execution) models using the same interface.

#### Examples

The command has many optional training-related parameters commonly tuned by the 
experimenter, like `batch-size`, `epochs`, `lr`. 

The most basic training can be performed specifying just the directory containing
the dataset, already split in `train` (compulsory) and `dev` (optional) folders
using the default values for the other parameters.
```
python3 -m tfkeras_template train data/npz
```

An equivalent form of the previous command with all the default values
manually specified is:
```
python3 -m tfkeras_template train \
    data/npz \
    --output-dir ./runs \
    --batch-size 20 \
    --epochs 30 \
    --lr 0.1
```

An optional flag `--imperative` allows to force the training to
use the imperative version of the model. The default model is based
on a symbolic computational graph, the fastest and most optimized way to define a model. 

The same command allows to restore a previous training from a checkpoint
```
python3 -m tfkeras_template train \
    data/npz \
    --output-dir ./runs \
    --checkpoint ./runs/000000-symbolic/checkpoints/checkpoint-10-0.10
```

A Tensorflow checkpoint is composed by a file with extension `.index`
and one or more files with suffix `.data-<part>-of-<total>`.
The correct checkpoint name to be passed is without any extension.
For example, the previous command works if the checkpoint directory 
contains a pair of files named `checkpoint-10-0.10.index` and
`checkpoint-10-0.10.data-000000-of-000001`.

For more details on the usage you can access the help page with the command
```
python3 -m tfkeras_template eval --help
```

### Command `eval`

The `eval` command reproduces the validation performed at the end of every epoch during 
the training phase. It is particularly useful when many datasets are available to
evaluate the transfer learning performances.

#### Examples

The evaluation can be performed specifying just the model checkpoint
to be evaluated and the directory containing the dataset, provided of a `dev` sub-folder.

The command can be called with:
```
python3 -m tfkeras_template eval \
    ./runs/000000-symbolic/checkpoints/checkpoint-10-0.10 \
    data/npz \
```

It is necessary to add the flag `--impartive` if the model was defined in
an imperative way during training. 

For more details on the usage you can access the help page with the command
```
python3 -m tfkeras_template eval --help
```

### Command `test`

The `test` command preforms the inference on a single file.

#### Examples

The test of the model is performed specifying a model checkpoint
and the path to an input file.
For example:
```
python3 -m tfkeras_template test  \
    ./runs/000000-symbolic/checkpoints/checkpoint-10-0.10 \
    data/test/test_000.npy
```

If the checkpoint refers to a model defined imperatively,
it is necessary to use the flag `--imparative` to load it correctly.

For more details on the usage you can access the help page with the command
```
python3 -m tfkeras_template test --help
```

## Deployment

The template can be deployed on an NGC optimized instance, here we list
the steps necessary to configure it on a AWS EC2 **g4dn.xlarge** instance
on the **NVIDIA Deep Learning AMI** environment.

1. Log in via ssh following the instructions on the EC2 Management Dashboard.
2. Clone the repo `tfkeras-template` in the home directory.
3. Download the most update Tensorflow container running 
`docker pull nvcr.io/nvidia/tensorflow:YY.MM-tf2-py3`
4. Create a container with
```
docker run --gpus all --name template -e HOME=$HOME -e USER=$USER \
    -v $HOME:$HOME -p 6006:6006 --shm-size 60G -it nvcr.io/nvidia/tensorflow:YY.MM-tf2-py3
```
At the end of the procedure you will gain access to a terminal on a Docker
container configured to work on the GPU and you could simply run the commands
above leveraging the speed of parallel computing.

The `$HOME` directory on the Docker container is linked to the `$HOME` directory
of the host machine, so the repository can be found in the `$HOME`, similarly the
port 6006 used by TensorBoard is remapped from the container to the port 6006
of the host machine.

Useful commands to interact with the Docker container are:
- `docker start template`: start the container;
- `docker exec -it template bash`: open a terminal on the container;
- `docker stop template`: stop the container;
- `docker rm template`: remove the container.

In order to monitor training you can run the following commands from the container console:
- `watch -n 1 nvidia-smi` to monitor GPU usage;
- `tensorboard --logdir runs/<run_id> --bind_all` to start Tensorboard.

## License

This project is licensed under Apache License 2.0,
see the [LICENSE](LICENSE) file for details.