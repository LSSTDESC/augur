# Augur

Augur is a DESC forecasting and inference validation tool. The name comes from the official diviners in ancient Rome whose function was to divine whether the gods approved of a proposed undertaking by observing the behavior of birds. Firecrown is the bird species of choice in DESC.

Augur is a wrapper to firecrown that generates synthetic datasets of abitrary complexity and then calls inference engine to either generate full MCMC or a simple second-order derivative at fiducal model to generate a Fisher matrix forecast.

## Installation

As always, you can force installation through pip like

```pip install augur/```

or actually inside the augur directory running

```python setup.py install```

## Step-by-Step Installation



## Usage

Usage generally follows the firecrown conventions. The input yaml file has three sections corresponding to three steps of a typical forecasting process
 * `generate` contains instructions for generating syntehtic datasets
 * `analyze` contains instructions for running firecrown using the dasets just generates
 * `postprocess` contains instructions for post-processing any data, making plots, latex tables, etc [not implemented yet]
 
 To run something try:
 ```
 augur examples/srd_v1_3x2.yaml -v
 ```
