import logging
import pytest
import numpy as np
import pyccl as ccl
import sacc

from augur.generate import _get_tracers, _get_scale_cuts, _build_tp_filters_from_sacc


# ---------------------------------------------------------------------------
# Helpers shared by _build_tp_filters_from_sacc tests
# ---------------------------------------------------------------------------

def _make_sacc_shear():
    """Minimal sacc with two shear tracers and two EE combinations."""
    S = sacc.Sacc()
    z = np.linspace(0.1, 2.0, 20)
    nz0 = np.exp(-0.5 * ((z - 0.5) / 0.2) ** 2)
    nz1 = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
    S.add_tracer('NZ', 'src0', z, nz0)
    S.add_tracer('NZ', 'src1', z, nz1)
    ells = np.array([10., 50., 100., 200., 500., 1000.])
    cls = np.zeros(len(ells))
    S.add_ell_cl('galaxy_shear_cl_ee', 'src0', 'src1', ells, cls)
    S.add_ell_cl('galaxy_shear_cl_ee', 'src0', 'src0', ells, cls)
    return S


def _make_cosmo():
    return ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)


def test_get_tracers_galaxy_density():
    assert _get_tracers('galaxy_density_cl', (0, 1)) == ('lens0', 'lens1')


def test_get_tracers_shear_ee():
    assert _get_tracers('galaxy_shear_cl_ee', (2, 3)) == ('src2', 'src3')


def test_get_tracers_shear_density():
    assert _get_tracers('galaxy_shearDensity_cl_e', (1, 4)) == ('src4', 'lens1')


def test_get_tracers_unrecognized():
    with pytest.raises(NotImplementedError):
        _get_tracers('unknown_stat', (0, 1))


def test_get_scale_cuts_scalar_lmax():
    stat_cfg = {'lmax': 500, 'tracer_combs': [[0, 0], [1, 1]]}
    lmax, kmax = _get_scale_cuts(stat_cfg, [0, 0])
    assert lmax == 500
    assert kmax is None


def test_get_scale_cuts_list_lmax():
    stat_cfg = {'lmax': [100, 200], 'tracer_combs': [[0, 0], [1, 1]]}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, (1, 1))
    lmax, _ = _get_scale_cuts(stat_cfg, [1, 1])
    assert lmax == 200


def test_get_scale_cuts_list_length_mismatch():
    stat_cfg = {'lmax': [100], 'tracer_combs': [[0, 0], [1, 1]]}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, (1, 1))


def test_get_scale_cuts_both_kmax_lmax():
    stat_cfg = {'lmax': 100, 'kmax': 0.2}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, [0, 0])


def test_get_scale_cuts_invalid_kmax_type():
    stat_cfg = {'kmax': 'not_a_number'}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, [0, 0])


# ---------------------------------------------------------------------------
# Tests for _build_tp_filters_from_sacc
# ---------------------------------------------------------------------------

# _make_sacc_shear() contains two combinations: (src0, src1) and (src0, src0).
# _build_tp_filters_from_sacc returns two kinds of filters:
#   * "real" filters (interval floored at 0.0) for configured combos present in
#     the sacc; and
#   * "exclusion" filters (interval placed just above the last ell, so no data
#     survives) for sacc combos that are NOT listed in the config, which drops
#     them from the likelihood.
# The last ell in the test sacc is 1000, so exclusion intervals are (1001, 1002).

def _real_filters(filters):
    """Filters that actually keep data: lower bound floored at 0.0."""
    return [f for f in filters if f.interval[0] == 0.0]


def _exclusion_filters(filters):
    """Drop filters for unlisted sacc combos: lower bound above the data range."""
    return [f for f in filters if f.interval[0] > 0.0]


class TestBuildTpFiltersFromSacc:

    def test_no_scale_cuts_returns_one_filter_per_comb(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        assert len(_real_filters(filters)) == 1

    def test_no_scale_cuts_cut_high_is_last_ell(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # cut_low is floored at 0.0 so the lowest bin's window is fully retained.
        assert _real_filters(filters)[0].interval == (0.0, 1000.0)

    def test_lmax_sets_cut_high(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'lmax': 300.0}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        real = _real_filters(filters)
        assert len(real) == 1
        assert real[0].interval[1] == pytest.approx(300.0)

    def test_ignore_sc_likelihood_overrides_lmax(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'lmax': 300.0}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=True)
        real = _real_filters(filters)
        assert len(real) == 1
        # lmax is ignored; cut_high should be last ell in sacc
        assert real[0].interval[1] == pytest.approx(1000.0)

    def test_kmax_produces_finite_cut_high(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'kmax': 0.2}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        real = _real_filters(filters)
        assert len(real) == 1
        cut_high = real[0].interval[1]
        assert np.isfinite(cut_high)
        assert cut_high > 0.0

    def test_ignore_sc_likelihood_overrides_kmax(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'kmax': 0.2}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=True)
        real = _real_filters(filters)
        assert len(real) == 1
        assert real[0].interval[1] == pytest.approx(1000.0)

    def test_multiple_combs_returns_multiple_filters(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1], [0, 0]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # Both sacc combos are configured -> two real filters, no exclusions.
        assert len(_real_filters(filters)) == 2
        assert _exclusion_filters(filters) == []

    def test_unlisted_sacc_combo_gets_exclusion_filter(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        # Only (src0, src1) configured; (src0, src0) present in sacc must be dropped.
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        excl = _exclusion_filters(filters)
        assert len(excl) == 1
        # Exclusion interval sits above the last ell so no data survives it.
        assert excl[0].interval[0] > 1000.0

    def test_missing_combo_skipped_with_warning(self, caplog):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        # src5, src6 not in sacc
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[5, 6]]}}
        with caplog.at_level(logging.WARNING, logger='augur.generate'):
            filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # No configured combo is present -> no real filters (only exclusions).
        assert _real_filters(filters) == []
        assert 'not found in sacc' in caplog.text

    def test_present_and_missing_combo_mixed(self, caplog):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1], [5, 6]]}}
        with caplog.at_level(logging.WARNING, logger='augur.generate'):
            filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # Only the present combination produces a real filter
        assert len(_real_filters(filters)) == 1
        assert 'not found in sacc' in caplog.text

    def test_cut_low_is_zero(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # cut_low is floored at 0.0 (not the first ell) so the lowest bin's full
        # bandpower window is retained under the SUPPORT filter method.
        assert _real_filters(filters)[0].interval[0] == 0.0

    def test_returns_empty_list_when_no_valid_combs(self, caplog):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[9, 9]]}}
        with caplog.at_level(logging.WARNING, logger='augur.generate'):
            filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # No configured combo is present -> no real filters.
        assert _real_filters(filters) == []
