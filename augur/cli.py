"""Command-line interface for Augur."""

import logging
import pprint

import click

import augur
from augur.analyze import Analyze
from augur.generate import generate
from augur.postprocess import postprocess

logger = logging.getLogger(__name__)


@click.command()
@click.argument("config", type=str)
@click.option("-v", "--verbose", is_flag=True)
def run(config, verbose):
    """Run Augur from a YAML configuration file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    parsed_config = augur.parse(config)
    if verbose:
        logger.debug("rendered config file:\n%s", pprint.pformat(parsed_config))

    logger.info("Generating fiducial data vector...")
    likelihood, S, tools, req_params = generate(parsed_config, return_all_outputs=True)

    if "fisher" in parsed_config:
        logger.info("Computing Fisher matrix...")
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
        logger.info("Postprocessing...")
        postprocess(parsed_config)

    logger.info("Done.")


if __name__ == "__main__":
    run()
