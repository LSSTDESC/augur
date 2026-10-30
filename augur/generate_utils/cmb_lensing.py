"""CMB lensing statistics for SACC generation.

This module adds harmonic-space CMB lensing auto- and cross-spectra to the
SACC data vector and builds matching firecrown ``TwoPoint`` statistics.

Originally written by Paul Rogozenski on the ``refactor`` branch; ported forward
onto ``master`` with the firecrown 1.16 import paths, a ``Map`` SACC tracer in
place of a delta-function ``NZ`` tracer, and ell-dependent reconstruction noise.
"""

import logging
import warnings

import numpy as np

from firecrown.likelihood import CMBConvergence, TwoPoint

from augur.utils.config_io import parse_array, CMB_Z_SOURCE_DEFAULT
from augur.utils.firecrown_interface import create_twopoint_filter

logger = logging.getLogger(__name__)

CMB_TRACER_NAME = 'cmb_convergence'

# Harmonic-space CMB lensing data types augur knows how to generate.
SUPPORTED_CMB_STATISTICS = {
    'cmb_convergence_cl',
    'cmbGalaxy_convergenceDensity_cl',
    'cmbGalaxy_convergenceShear_cl_e',
}

# Number of galaxy-bin indices each statistic expects in a `tracer_combs` entry.
# kappa is implicit in every pair, so only the galaxy bin is named.
_COMB_ARITY = {
    'cmb_convergence_cl': 0,
    'cmbGalaxy_convergenceDensity_cl': 1,
    'cmbGalaxy_convergenceShear_cl_e': 1,
}

NOISE_CONVENTIONS = ('kappa', 'phi', 'dd')


def _convert_noise_convention(ells, n_ell, convention):
    """Convert a reconstruction-noise curve to the convergence convention.

    The lensing potential, the deflection field and the convergence are related
    by ``kappa_lm = -L(L+1)/2 phi_lm`` and ``d_lm = sqrt(L(L+1)) phi_lm``, hence

    ``N_L^kk = [L(L+1)/2]^2 N_L^pp`` and ``N_L^kk = [L(L+1)/4] N_L^dd``.

    Parameters
    ----------
    ells : np.ndarray
        Multipoles at which ``n_ell`` is tabulated.
    n_ell : np.ndarray
        Noise curve in the given convention.
    convention : str
        One of ``'kappa'``, ``'phi'`` or ``'dd'``.

    Returns
    -------
    np.ndarray
        ``N_L^{kappa kappa}``.
    """
    if convention not in NOISE_CONVENTIONS:
        raise ValueError(
            f"Unknown CMB lensing noise convention '{convention}'. "
            f"Supported conventions are: {sorted(NOISE_CONVENTIONS)}"
        )
    if convention == 'kappa':
        return np.asarray(n_ell, dtype=float)

    ells = np.asarray(ells, dtype=float)
    fac = ells * (ells + 1.0)
    if convention == 'phi':
        return np.asarray(n_ell, dtype=float) * (fac / 2.0) ** 2
    # convention == 'dd'
    return np.asarray(n_ell, dtype=float) * fac / 4.0


def get_cmb_noise(cmb_cfg, ells):
    """Return the CMB convergence reconstruction noise evaluated at ``ells``.

    Two config forms are accepted. A scalar white-noise level::

        cmb_lensing:
            noise_cl: 1.0e-8

    or a tabulated curve, which is what a real forecast needs::

        cmb_lensing:
            noise:
                file: nlkk_so_lat_baseline.dat
                ell_col: 0
                nl_col: 7
                convention: kappa      # kappa | phi | dd

    The curve is interpolated in log-log space onto ``ells``. Outside the
    tabulated reconstruction band the noise is ``inf``, not zero, so that those
    modes carry no weight in the covariance -- a zero there would silently make
    the reconstruction look noiseless.

    Parameters
    ----------
    cmb_cfg : dict
        The ``cmb_lensing`` config section.
    ells : array_like
        Multipoles to evaluate the noise at.

    Returns
    -------
    np.ndarray
        ``N_L^{kappa kappa}`` with the same shape as ``ells``.
    """
    ells = np.atleast_1d(np.asarray(ells, dtype=float))

    noise_cfg = cmb_cfg.get('noise', None)
    if noise_cfg is None:
        return np.full_like(ells, float(cmb_cfg.get('noise_cl', 0.0)))

    if 'noise_cl' in cmb_cfg:
        warnings.warn(
            "Both `cmb_lensing.noise` and `cmb_lensing.noise_cl` are set; "
            "using the tabulated `noise` curve and ignoring `noise_cl`."
        )

    path = noise_cfg['file']
    table = np.loadtxt(path)
    if table.ndim != 2:
        raise ValueError(
            f"CMB lensing noise file '{path}' must be a 2D table of columns, "
            f"got an array of shape {table.shape}."
        )
    ell_col = int(noise_cfg.get('ell_col', 0))
    nl_col = int(noise_cfg.get('nl_col', 1))
    for name, col in (('ell_col', ell_col), ('nl_col', nl_col)):
        if col >= table.shape[1]:
            raise ValueError(
                f"CMB lensing noise file '{path}' has {table.shape[1]} columns, "
                f"but {name}={col} was requested."
            )

    ell_tab = table[:, ell_col]
    nl_tab = _convert_noise_convention(
        ell_tab, table[:, nl_col], noise_cfg.get('convention', 'kappa')
    )

    good = np.isfinite(nl_tab) & (nl_tab > 0) & (ell_tab > 0)
    if not np.any(good):
        raise ValueError(
            f"CMB lensing noise file '{path}' contains no positive, finite "
            "noise values after conversion."
        )
    ell_tab, nl_tab = ell_tab[good], nl_tab[good]

    # Interpolate the curve itself rather than indexing by row: these tables are
    # (L, N_L) pairs and their row index is not the multipole.
    logger.info(
        "CMB lensing noise read from %s; reconstruction band L = [%g, %g]",
        path, ell_tab[0], ell_tab[-1],
    )
    n_ell = np.exp(np.interp(np.log(ells), np.log(ell_tab), np.log(nl_tab)))

    # inf outside the band, so those modes are given no weight.
    outside = (ells < ell_tab[0]) | (ells > ell_tab[-1])
    if np.any(outside):
        warnings.warn(
            f"{int(np.sum(outside))} of {ells.size} requested multipoles fall "
            f"outside the reconstruction band [{ell_tab[0]:g}, {ell_tab[-1]:g}] "
            f"of '{path}'; their noise is set to inf."
        )
        n_ell = np.where(outside, np.inf, n_ell)
    return n_ell


def add_cmb_tracer(S, sources, z_source, lmax=None):
    """Register the CMB convergence tracer in the SACC file and as a source.

    The tracer is written as a SACC ``Map`` tracer carrying **both**
    ``quantity="cmb_convergence"``, which is what TJPCov dispatches on, and
    ``metadata={"z_lss": z_source}``, which is what firecrown reads back. The
    two conventions coexist and both consumers need their own.

    Using a ``Map`` tracer rather than a delta-function ``NZ`` tracer matters:
    firecrown dispatches on the SACC tracer *class*
    (``firecrown.metadata_functions``), so an ``NZ`` kappa tracer is read back
    as a tomographic galaxy bin.
    """
    if CMB_TRACER_NAME not in S.tracers:
        beam_ells = np.array([0.0, float(lmax) if lmax is not None else 1.0e5])
        S.add_tracer(
            'Map', CMB_TRACER_NAME, 0, beam_ells, np.ones_like(beam_ells),
            quantity='cmb_convergence',
            metadata={'z_lss': float(z_source)},
        )
    if CMB_TRACER_NAME not in sources:
        sources[CMB_TRACER_NAME] = CMBConvergence(
            sacc_tracer=CMB_TRACER_NAME, z_source=float(z_source),
        )
    return sources[CMB_TRACER_NAME]


def _tracer_pair(key, comb):
    """Return the ordered (tracer0, tracer1) names for a CMB statistic.

    kappa is always the *first* tracer: firecrown orders measurements
    ``CMB < Clusters < Galaxies`` and putting kappa second triggers its
    deprecated auto-swap path.
    """
    arity = _COMB_ARITY[key]
    if len(comb) != arity:
        if arity == 0:
            raise ValueError(
                f"For {key} use empty tracer combinations: tracer_combs: [[]]"
            )
        raise ValueError(
            f"For {key} use single-bin combos, e.g. [[0], [1], ...]; "
            f"got {comb!r}"
        )
    if key == 'cmb_convergence_cl':
        return CMB_TRACER_NAME, CMB_TRACER_NAME
    if key == 'cmbGalaxy_convergenceDensity_cl':
        return CMB_TRACER_NAME, f'lens{comb[0]}'
    return CMB_TRACER_NAME, f'src{comb[0]}'


def add_cmb_lensing(config, S, sources, dndz, cosmo):
    """Populate a SACC object with harmonic CMB-lensing data points.

    Parameters
    ----------
    config : dict
        Full Augur config dict (should contain ``cmb_lensing``).
    S : sacc.Sacc
        SACC object to populate.
    sources : dict
        Tracer-name -> firecrown source mapping (updated in-place when the
        CMB-lensing tracer is added).
    dndz : dict
        Tracer-name -> N(z) mapping (unused for CMB lensing, which has no
        redshift distribution, but kept for API consistency).
    cosmo : pyccl.Cosmology
        Fiducial cosmology. Unused today; kept so that scale cuts derived from
        the cosmology can be added without changing the signature.

    Returns
    -------
    stats : list
        Firecrown TwoPoint statistics for the CMB auto/cross terms.
    tp_filters : list
        TwoPoint filters encoding the ell scale cuts.
    """
    cmb_cfg = config.get('cmb_lensing', None)
    if cmb_cfg is None:
        return [], []

    ignore_sc = config['general'].get('ignore_scale_cuts', False)
    ignore_sc_likelihood = config['general'].get('ignore_scale_cuts_likelihood', False)
    if not ignore_sc and ignore_sc_likelihood:
        raise ValueError(
            "Cannot ignore scale cuts in likelihood while "
            "applying them to the data vector."
        )

    stat_keys = cmb_cfg.get('statistics', {})
    for key in stat_keys:
        if key not in SUPPORTED_CMB_STATISTICS:
            raise NotImplementedError(
                f"CMB lensing statistic '{key}' is not supported in Augur yet. "
                f"Supported keys are: {sorted(SUPPORTED_CMB_STATISTICS)}"
            )

    z_source = cmb_cfg.get('z_source', CMB_Z_SOURCE_DEFAULT)
    lmax_overall = max(
        (float(cfg['lmax']) for cfg in stat_keys.values()
         if cfg.get('lmax', None) not in (None, 'None')),
        default=None,
    )
    add_cmb_tracer(S, sources, z_source, lmax=lmax_overall)

    stats = []
    tp_filters = []

    for key in stat_keys:
        tracer_combs = stat_keys[key].get('tracer_combs', [])
        ell_edges = parse_array(stat_keys[key]['ell_edges'])
        ells = np.sqrt(ell_edges[:-1] * ell_edges[1:])  # Geometric average
        lmax_cfg = stat_keys[key].get('lmax', None)

        for comb in tracer_combs:
            tr1, tr2 = _tracer_pair(key, comb)
            if tr2 not in sources:
                raise ValueError(
                    f"Tracer '{tr2}' needed for '{key}' is missing. "
                    "Ensure the matching sources/lenses section is defined."
                )

            # kappa has no n(z), so the kmax -> lmax conversion used for galaxy
            # tracers cannot apply here; the reconstruction band is config only.
            ells_here = ells
            if lmax_cfg is not None and lmax_cfg != 'None':
                lmax = float(lmax_cfg)
                ells_here = ells[ells <= lmax]
                if len(ells_here) == 0:
                    raise ValueError(
                        f"lmax={lmax} removes all ell bins for {key} ({tr1}, {tr2})."
                    )

            if not ignore_sc_likelihood:
                tp_filters.append(
                    create_twopoint_filter(
                        key, tr1, tr2,
                        cut_low=float(ells_here[0]),
                        cut_high=float(ells_here[-1]),
                    )
                )

            if ignore_sc:
                ells_here = ells

            S.add_ell_cl(key, tr1, tr2, ells_here, np.zeros(len(ells_here)))
            stats.append(
                TwoPoint(
                    source0=sources[tr1],
                    source1=sources[tr2],
                    sacc_data_type=key,
                )
            )

    if len(stats) == 0:
        warnings.warn(
            "cmb_lensing section is present but produced no statistics. "
            "Check cmb_lensing.statistics and tracer_combs entries."
        )

    return stats, tp_filters
