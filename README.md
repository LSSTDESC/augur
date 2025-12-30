![](https://github.com/LSSTDESC/augur/workflows/flake8%20pytest/badge.svg)

# Augur

Augur is a DESC forecasting and inference validation tool. The name comes from the official diviners in ancient Rome whose function was to divine whether the gods approved of a proposed undertaking by observing the behavior of birds. Firecrown is the bird species of choice in DESC.

Augur is a wrapper to firecrown that generates synthetic datasets of abitrary complexity and then calls inference engine to either generate full MCMC or a simple second-order derivative at fiducal model to generate a Fisher matrix forecast.

## Installation

As always, you can force installation through pip like

```pip install augur/```

or actually inside the augur directory running

```python setup.py install```

## Step-by-Step Installation (via conda)

This step-by-step installaion shows you how to get a working environment with `firecrown` and `augur` that you can hack away efficiently.

### Clone the repository
First clone and enter the `augur` repository:
```bash
git clone git@github.com:LSSTDESC/augur.git
cd augur
# set the variable AUGUR_DIR to the current directory
AUGUR_DIR=${PWD}
```

### Create a new conda environment
Start by creating a new anaconda environment from the `environment.yml` file in the `augur` directory:

```bash
conda env create --name forecasting --file=environment.yml
conda activate forecasting
```
### Or updating an existing environment
If you already have an existing environment, you can update it with the following command:

```bash
conda activate my_env
conda env update --name my_env --file=environment.yml --prune
```
and activate your environment with `conda activate [my_env/forecasting]`.

### Cosmosis setup
Now we need to build the cosmosis standard library:
```bash
source ${CONDA_PREFIX}/bin/cosmosis-configure
cd ${CONDA_PREFIX}
cosmosis-build-standard-library
# go back to the augur directory
cd ${AUGUR_DIR}
```

### Configure paths necessary for Augur
We need to let augur know about the location of Firecrown, to do that, run the following command:
```bash
FIRECROWN_DIR=$(python -c "import firecrown; print('/'.join(firecrown.__spec__.submodule_search_locations[0].split('/')[0:-1]))")
```
Now let's set the environment variables that augur needs to know about:
```bash
conda env config vars set AUGUR_DIR=${PWD} CSL_DIR=${CONDA_PREFIX}/cosmosis-standard-library FIRECROWN_DIR=${FIRECROWN_DIR}
```
we now need to reload our environment to make the changes effective:
```bash
conda deactivate
conda activate forecasting
```

### Finally install augur
```bash
cd ${AUGUR_DIR}
# now install augur via pip in editable mode
python -m pip install --no-deps --editable ${PWD}
```

<!-- Next install firecrown and augur.

Install a repo version of firecrown:

```
git clone git@github.com:LSSTDESC/firecrown.git
cd firecrown
pythong -m pip install .
```

Now run a `pytest` to see if things work.

Next repeat the same with `augur`: -->

#### Using developer version of Firecrown
To install and modify the developer version of `firecrown`, follow the instructions [here](https://firecrown.readthedocs.io/en/latest/developer_installation.html).

-----------
## Installation using Conda (all in one place)

Because Augur depends upon Firecrown, and Firecrown requires installation using Conda, we use Conda to install Augur.
```bash
# clone the Augur repository
git clone git@github.com:LSSTDESC/augur.git

cd augur

# conda env update, when run as suggested, is able to create a new environment, as
# well as updating an existing environment.
conda env update -f environment.yml
conda activate forecasting

# The following line loads the firecrown module from the environment, and queries
# it to find the installation location.
FIRECROWN_DIR=$(python -c "import firecrown; print('/'.join(firecrown.__spec__.submodule_search_locations[0].split('/')[0:-1]))")

# We define some environment variables that will be defined whenever you activate
# the conda environment.
conda env config vars set AUGUR_DIR=${PWD} CSL_DIR=${CONDA_PREFIX}/cosmosis-standard-library FIRECROWN_DIR=${FIRECROWN_DIR}
# The command above does not immediately define the environment variables.
# They are made available on every fresh activation of the environment.
# So we have to deactivate and then reactivate...
conda deactivate
conda activate forecasting
# Now we can finish building the CosmoSIS Standard Library.
source ${CONDA_PREFIX}/bin/cosmosis-configure
# We want to put the CSL into the same directory as conda environment upon which it depends
cd ${CONDA_PREFIX}
cosmosis-build-standard-library
# Now change directory into the Augur repository
cd ${AUGUR_DIR}
# And finally make an editable (developer) installation of firecrown into the conda environment
python -m pip install --no-deps --editable ${PWD}
```

## Usage

`augur` has changed from its initial version and currently only contains a
`generate` stage where it creates the firecrown-likelihood object and fiducial data vector as specified by the configuration file passed to this stage.

The user can create configuration files to fit their specific purposes following the example configuration files in the [`examples`](./examples) subdirectory. We show a quick example of how to obtain a likelihood object from an example configuration file.

```
from augur.generate import generate
lk = generate('./examples/config_test.yml', return_all_outputs=False)
```

This likelihood object can then be used by `cosmosis`, `cobaya` or `NumCosmo`. For more details follow the examples in the [`firecrown`](https://github.com/LSSTDESC/firecrown) repository.

Additionally, we can compute the Fisher matrix and Fisher biases via numerical derivatives using the following commands:

```
from augur.analyze import Analyze
ao = Analyze('./examples/config_test.yml')
ao.get_fisher_bias(method='5pt_stencil')  # This command computes the derivates+Fisher matrices+fisher bias
print(ao.Fij, ao.bi)  # These are the values of the Fisher matrix, Fij, and Fisher biases bi
```

## Example run for SRD v1
We also include example configuration files for `cosmosis` and `cobaya` to reproduce the results from the [LSST DESC Science Requirements Document](https://arxiv.org/pdf/1809.01669.pdf).

__Note: It is left to the discretion of the user which inference framework to use. In order to run the chains the user can decide whether they want to install `cosmosis`, `cobaya`, or `NumCosmo`, or a combination of them.__

### Cosmosis

First, you will need to have `cosmosis-standard-library` installed. Follow the instructions [here](https://cosmosis.readthedocs.io/en/latest/intro/installation.html) to install cosmosis.

To run the example chain you can just do:

```
cd examples
cosmosis srd_y1_3x2_cosmosis.ini
```

By default, the results from this run will be saved at:

`${AUGUR_DIR}/output/SRD_y1_3x2pt_samples.txt`

Feel free to modify the paths indicated in this configuration file as needed or define the following environment variables:

* `CSL_DIR`: The path to the directory where the cosmosis standard library can be found. It should contain the cosmosis modules.
* `AUGUR_DIR`: The directory where you have `augur` or the `augur` installation.
* `FIRECROWN_DIR`: The directory where you have `firecrown` or the `firecrown` installation.

### Cobaya

Similarly to the case of `cosmosis`, first you need to have `cobaya` installed in order to be able to run the `cobaya` example. Please follow the instructions [here](https://cobaya.readthedocs.io/en/latest/installation.html) to learn how to install `cobaya`.

To run the example chain you can just do:

```
cd examples
cobaya-run cobaya_mcmc.yaml
```

By default the outputs will be saved at `./examples/cobaya_evaluate_output`.
