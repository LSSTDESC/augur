import os
import warnings

import healpy as hp
import numpy as np
import pyccl as ccl
import pytest
import sacc

pytest.importorskip('pymaster')

from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt  # noqa: E402

from augur.generate_utils import namaster as nm  # noqa: E402
from augur.utils.cov_utils import AugurTracerInfo  # noqa: E402
from augur.utils.tjpcov_nmt import TJPCovNmt  # noqa: E402

NSIDE = 16
LMAX = 3 * NSIDE - 1
EDGES = [2, 10, 20, 30, 48]


def _cap(theta_max_deg):
    theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    return (np.degrees(theta) < theta_max_deg).astype(float)


def _sacc(nbands=len(EDGES) - 1):
    S = sacc.Sacc()
    z = np.linspace(0.01, 2.0, 100)
    S.add_tracer('NZ', 'src0', z, np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2), quantity='galaxy_shear')
    S.add_tracer('NZ', 'lens0', z, np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2),
                 quantity='galaxy_density')
    S.add_tracer('Map', 'cmb_convergence', 0, np.array([0.0, 50.0]), np.ones(2),
                 quantity='cmb_convergence')
    ells = np.linspace(5, 40, len(EDGES) - 1)
    for dtype, t1, t2 in [('galaxy_shear_cl_ee', 'src0', 'src0'),
                          ('galaxy_shear_cl_bb', 'src0', 'src0'),
                          ('galaxy_shearDensity_cl_e', 'src0', 'lens0'),
                          ('galaxy_density_cl', 'lens0', 'lens0'),
                          ('cmbGalaxy_convergenceDensity_cl', 'cmb_convergence', 'lens0'),
                          ('cmb_convergence_cl', 'cmb_convergence', 'cmb_convergence')]:
        n = nbands if dtype == 'galaxy_density_cl' else len(ells)
        S.add_ell_cl(dtype, t1, t2, ells[:n], np.zeros(n))
    return S


def _tjpcov_config(S, **extra):
    tj = {'sacc_file': S, 'cosmo': ccl.CosmologyVanillaLCDM(), 'IA': None,
          'Ngal_src0': 10.0, 'sigma_e_src0': 0.26, 'Ngal_lens0': 5.0, 'bias_lens0': 1.5,
          'cmb_noise': 1e-7}
    tj.update(extra)
    return {'tjpcov': tj}


@pytest.fixture
def namaster(tmp_path):
    files = {'wide': tmp_path / 'wide.fits', 'small': tmp_path / 'small.fits'}
    hp.write_map(str(files['wide']), _cap(100), dtype=np.float64)
    hp.write_map(str(files['small']), _cap(60), dtype=np.float64)
    return nm.load_namaster({'surveys': {
        'footprint': {'masks': {k: {'file': str(v)} for k, v in files.items()},
                      'namaster': {}},
        'sources': {'mask': 'wide'}, 'lenses': {'mask': 'wide'},
        'cmb_lensing': {'mask': 'small'},
    }})


class _Reference(AugurTracerInfo, FourierGaussianNmt):
    """Upstream TJPCov NaMaster class, building its own fields and workspaces."""


def _pairs(S):
    combs = S.get_tracer_combinations()
    return [(combs[i], combs[j]) for i in range(len(combs)) for j in range(i, len(combs))]


def test_matches_upstream_tjpcov(tmp_path, namaster):
    S = _sacc()
    names = {'src0': 'wide', 'lens0': 'wide', 'cmb_convergence': 'small'}
    ref = _Reference(_tjpcov_config(
        S, mask_names=names, outdir=str(tmp_path / 'ref'),
        mask_file={t: _cap(100 if n == 'wide' else 60) for t, n in names.items()},
        binning_info=nm.make_nmt_bin(EDGES, LMAX),
    ))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with TJPCovNmt(_tjpcov_config(S), namaster, EDGES) as calc:
            outdir = calc.io.outdir
            calc.precompute(_pairs(S))
            for comb1, comb2 in _pairs(S):
                mine = calc.get_covariance_block_for_sacc(comb1, comb2)
                theirs = ref.get_covariance_block_for_sacc(comb1, comb2, use_coupled_noise=False)
                assert mine.shape == theirs.shape
                assert np.allclose(mine, theirs, rtol=1e-12, atol=0), (comb1, comb2)
            shear = calc.get_covariance_block_for_sacc(('src0', 'src0'), ('src0', 'src0'))
            written = os.listdir(outdir)

    # E and B of src0 x src0 are both present, and BB-BB is non-zero.
    nb = len(EDGES) - 1
    assert shear.shape == (2 * nb, 2 * nb)
    assert np.all(np.diag(shear)[nb:] > 0)

    # Only TJPCov's block files were written (no workspace files), and they are gone.
    assert written and all(f.startswith('cov_') for f in written)
    assert not os.path.exists(outdir)
    assert set(namaster.origin.values()) == {'computed'}


def test_given_outdir_is_kept(tmp_path, namaster):
    S = _sacc()
    calc = TJPCovNmt(_tjpcov_config(S, outdir=str(tmp_path / 'keep')), namaster, EDGES)
    calc.cleanup()
    assert os.path.isdir(tmp_path / 'keep')


def test_missing_bands_raise(namaster):
    S = _sacc(nbands=len(EDGES) - 2)
    with pytest.raises(ValueError, match='every band'):
        TJPCovNmt(_tjpcov_config(S), namaster, EDGES)
