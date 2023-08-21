![](https://github.com/LSSTDESC/augur/workflows/flake8%20pytest/badge.svg)

# Augur

Augur is a DESC forecasting and inference validation tool. The name comes from the official diviners in ancient Rome whose function was to divine whether the gods approved of a proposed undertaking by observing the behavior of birds. Firecrown is the bird species of choice in DESC.

Augur is a wrapper to firecrown that generates synthetic datasets of abitrary complexity and then calls inference engine to either generate full MCMC or a simple second-order derivative at fiducal model to generate a Fisher matrix forecast.

## Installation

As always, you can force installation through pip like

```pip install augur/```

or actually inside the augur directory running

```python setup.py install```

## Step-by-Step Installation

This step-by-step installaion shows you how to get a working environment with `firecrown` and `augur` that you can hack away efficiently.

Start by creating a new anaconda environment:

```
conda create --name forecasting
conda activate forecasting
```

Next install firecrown and augur.

Install a repo version of firecrown:

```
git clone git@github.com:LSSTDESC/firecrown.git
cd firecrown
pip install --no-deps -e .
```

Now run a `pytest` to see if things work.

Next repeat the same with `augur` but checkout the `dev` branch:

```
git clone git@github.com:LSSTDESC/augur.git
cd augur
pip install --no-deps -e .
```

You are now ready to try a simple forecast as outlined in the next section.

## Usage

`augur` has changed from its initial version and currently only contains a
`generate` stage where it creates the firecrown-likelihood object and fiducial data vector as specified by the configuration file passed to this stage.

The user can create configuration files to fit their specific purposes following the example configuration files in the [`examples`](./examples) subdirectory. We show a quick example of how to obtain a likelihood object from an example configuration file.

```
from augur.generate import generate
lk = generate('./examples/config_test.yml', return_all_outputs=False, force_read=False)
```

This likelihood object can then be used by `cosmosis`, `cobaya` or `NumCosmo`. For more details follow the examples in the [`firecrown`](https://github.com/LSSTDESC/firecrown) repository.

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