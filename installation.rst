Installation
============

Augur depends on `firecrown`, which must be installed in a conda environment.
The recommended install path is to create a conda environment from `environment.yml`
and then install Augur in editable mode.

Requirements
------------
- `conda`
- `firecrown` installed via conda
- `cosmosis-standard-library` installed and built

Clone the repository
--------------------

```bash
git clone git@github.com:LSSTDESC/augur.git
cd augur
AUGUR_DIR=${PWD}
```

Create a new conda environment
------------------------------

```bash
conda env create --name forecasting --file environment.yml
conda activate forecasting
```

Or update an existing environment
----------------------------------

```bash
conda activate my_env
conda env update --name my_env --file environment.yml --prune
```

Build the Cosmosis standard library
-----------------------------------

```bash
source ${CONDA_PREFIX}/bin/cosmosis-configure
cd ${CONDA_PREFIX}
cosmosis-build-standard-library
cd ${AUGUR_DIR}
```

Configure environment variables
-------------------------------

```bash
FIRECROWN_DIR=$(python -c "import firecrown; print('/'.join(firecrown.__spec__.submodule_search_locations[0].split('/')[0:-1]))")
conda env config vars set AUGUR_DIR=${PWD} CSL_DIR=${CONDA_PREFIX}/cosmosis-standard-library FIRECROWN_DIR=${FIRECROWN_DIR}
conda deactivate
conda activate forecasting
```

Install Augur
-------------

```bash
cd ${AUGUR_DIR}
python -m pip install --no-deps --editable ${PWD}
```

Alternative pip installation
----------------------------

If `firecrown` is already installed in your conda environment, Augur may be
installed with:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

Legacy install methods
----------------------

These are not recommended, but may work if `firecrown` is already installed.

```bash
pip install augur/
python setup.py install
```

Developer Firecrown
-------------------

To install and modify a development version of `firecrown`, follow the
Firecrown developer installation instructions:

https://firecrown.readthedocs.io/en/latest/developer_installation.html
