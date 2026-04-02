"""Command-line interface for Augur."""

import pprint

import click

import augur
from augur.analyze import Analyze
from augur.generate import generate
from augur.postprocess import postprocess


@click.command()
@click.argument("config", type=str)
@click.option("-v", "--verbose", is_flag=True)
def run(config, verbose):
    """Run Augur from a YAML configuration file."""
    parsed_config = augur.parse(config)
    if verbose:
        print("rendered config file:\n", pprint.pformat(parsed_config), flush=True)

    print("Generating fiducial data vector...", flush=True)
    likelihood, S, tools, req_params = generate(parsed_config, return_all_outputs=True)

    if "fisher" in parsed_config:
        print("Computing Fisher matrix...", flush=True)
        analysis = Analyze(
            parsed_config,
            likelihood=likelihood,
            tools=tools,
            req_params=req_params,
        )
        analysis.get_fisher_matrix()
        if "fisher_bias" in parsed_config["fisher"]:
            analysis.get_fisher_bias()

    if "postprocess" in parsed_config:
        print("Postprocessing...", flush=True)
        postprocess(parsed_config)

    print("Done.", flush=True)


if __name__ == "__main__":
    run()
