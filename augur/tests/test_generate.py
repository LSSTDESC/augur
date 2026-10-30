from pathlib import Path

import pytest

from augur.generate import generate
from augur.utils.config_io import parse_config


def test_generate():
    base_path = Path(__file__).parent
    generate(f'{base_path}/test.yaml')


def test_generate_cmb_lensing():
    """6x2pt: 3x2pt plus the two kappa crosses and the kappa auto-spectrum."""
    base_path = Path(__file__).parent
    generate(f'{base_path}/test_cmb_lensing.yaml')


def _cmb_config(**overrides):
    """Load the tiny 6x2pt config as a dict so a test can mutate it."""
    base_path = Path(__file__).parent
    config = parse_config(f'{base_path}/test_cmb_lensing.yaml')
    config.update(overrides)
    return config


def test_generate_raises_on_z_source_mismatch(tmp_path):
    """Moving `cmb_lensing.z_source` without the firecrown factory must not be silent.

    Only `cmb_factories[].z_source` reaches the theory prediction, so this
    config would have built the data vector and the theory at different source
    redshifts with no error anywhere.
    """
    config = _cmb_config()
    config['fiducial_sacc_path'] = str(tmp_path / 'mismatch.sacc')
    config['cmb_lensing']['z_source'] = 950.0

    with pytest.raises(ValueError, match='cmb_factories') as exc_info:
        generate(config)
    msg = str(exc_info.value)
    assert '950.0' in msg and '1100.0' in msg


def test_generate_tjpcov_rejects_non_default_z_source(tmp_path):
    """TJPCov hard-codes z_source=1100 with no config hook, so refuse the combination.

    This raises before any cosmology or tracer is built, which is why the test
    is fast: TJPCov is never instantiated.
    """
    config = _cmb_config()
    config['fiducial_sacc_path'] = str(tmp_path / 'tjpcov.sacc')
    config['cmb_lensing']['z_source'] = 950.0
    config['Firecrown_Factory']['TwoPointFactory']['cmb_factories'][0]['z_source'] = 950.0
    config['cov_options']['cov_type'] = 'tjpcov'

    with pytest.raises(ValueError, match='124'):
        generate(config)


def test_generate_accepts_matching_non_default_z_source(tmp_path):
    """The guard must not narrow which configs augur accepts.

    A consistent non-default z_source is legitimate on the internal Gaussian
    path, where the covariance is built from the firecrown tracers.
    """
    config = _cmb_config()
    config['fiducial_sacc_path'] = str(tmp_path / 'consistent.sacc')
    config['cmb_lensing']['z_source'] = 950.0
    config['Firecrown_Factory']['TwoPointFactory']['cmb_factories'][0]['z_source'] = 950.0

    generate(config)
