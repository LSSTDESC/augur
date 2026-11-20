import importlib.util
import json
import os
import warnings

import healpy as hp
import numpy as np
import pytest
import sacc

from augur.generate_utils import namaster as nm

NSIDE = 16
LMAX = 3 * NSIDE - 1
EDGES = [2, 10, 20, 30, 48]

needs_pymaster = pytest.mark.skipif(
    importlib.util.find_spec('pymaster') is None, reason='pymaster not installed'
)


def _cap(nside, theta_max_deg):
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return (np.degrees(theta) < theta_max_deg).astype(float)


@pytest.fixture
def mask_files(tmp_path):
    files = {'wide': tmp_path / 'wide.fits', 'small': tmp_path / 'small.fits'}
    hp.write_map(str(files['wide']), _cap(NSIDE, 100), dtype=np.float64)
    hp.write_map(str(files['small']), _cap(NSIDE, 60), dtype=np.float64)
    return {k: str(v) for k, v in files.items()}


def _config(mask_files, namaster=None, **footprint_extra):
    footprint = {'masks': {'wide': {'file': mask_files['wide']},
                           'small': {'file': mask_files['small']}}}
    footprint.update(footprint_extra)
    if namaster is not None:
        footprint['namaster'] = namaster
    return {'surveys': {
        'footprint': footprint,
        'sources': {'mask': 'wide'},
        'lenses': {'mask': 'wide', 'bin_masks': {1: 'small'}},
        'cmb_lensing': {'mask': 'small'},
    }}


def _write_workspace(path, mask_a, spin_a, mask_b, spin_b):
    import pymaster as nmt
    bins = nm.make_nmt_bin(EDGES, LMAX)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(nmt.NmtField(mask_a, None, spin=spin_a),
                                nmt.NmtField(mask_b, None, spin=spin_b), bins)
    wsp.write_to(str(path))
    return wsp


# ---- footprint (healpy only) -------------------------------------------------

def test_tracer_probe():
    assert nm.tracer_probe('src3') == ('sources', 3)
    assert nm.tracer_probe('lens0') == ('lenses', 0)
    assert nm.tracer_probe('cmb_convergence') == ('cmb_lensing', None)
    with pytest.raises(ValueError, match='no NaMaster field'):
        nm.tracer_probe('cluster_z0_m0')


def test_no_surveys_section():
    assert nm.load_survey_footprint({}) is None
    assert nm.load_namaster({}) is None


def test_tracer_masks_and_spins(mask_files):
    fp = nm.load_survey_footprint(_config(mask_files))
    assert fp.tracer_mask_name('src0') == 'wide'
    assert fp.tracer_mask_name('lens0') == 'wide'
    assert fp.tracer_mask_name('lens1') == 'small'
    assert fp.tracer_mask_name('cmb_convergence') == 'small'
    assert fp.tracer_spin('src2') == 2
    assert fp.tracer_spin('lens1') == 0
    assert fp.tracer_spin('cmb_convergence') == 0
    assert fp.nside == NSIDE
    assert not fp.namaster_enabled


def test_mean_overlap(mask_files):
    fp = nm.load_survey_footprint(_config(mask_files))
    expected = np.mean(_cap(NSIDE, 100) * _cap(NSIDE, 60))
    assert fp.mean_overlap('wide', 'small') == pytest.approx(expected)


def test_missing_probe_entry(mask_files):
    cfg = _config(mask_files)
    del cfg['surveys']['cmb_lensing']
    fp = nm.load_survey_footprint(cfg)
    with pytest.raises(ValueError, match="no 'cmb_lensing' entry"):
        fp.tracer_mask_name('cmb_convergence')


@pytest.mark.parametrize('edit, match', [
    (lambda s: s.update(sorces={'mask': 'wide'}), 'Unknown key'),
    (lambda s: s['lenses'].update(mask='nope'), "mask 'nope'"),
    (lambda s: s['cmb_lensing'].update(bin_masks={0: 'wide'}), 'only allowed'),
    (lambda s: s['footprint']['masks']['wide'].update(apodize={'type': 'C1'}), 'scale_deg'),
    (lambda s: s['footprint']['masks']['wide'].update(
        apodize={'scale_deg': 5.0, 'type': 'C3'}), 'must be one of'),
])
def test_config_errors(mask_files, edit, match):
    cfg = _config(mask_files)
    edit(cfg['surveys'])
    with pytest.raises(ValueError, match=match):
        nm.load_survey_footprint(cfg)


def test_mask_cleaning(tmp_path, mask_files):
    arr = _cap(NSIDE, 100)
    arr[:3] = hp.UNSEEN
    np.save(tmp_path / 'unseen.npy', arr)
    cfg = _config(mask_files)
    cfg['surveys']['footprint']['masks']['wide'] = str(tmp_path / 'unseen.npy')
    fp = nm.load_survey_footprint(cfg)
    with pytest.warns(UserWarning, match='UNSEEN'):
        mask = fp.raw_mask('wide')
    assert np.all(mask[:3] == 0)

    arr[:3] = -1.0
    np.save(tmp_path / 'negative.npy', arr)
    cfg['surveys']['footprint']['masks']['wide'] = str(tmp_path / 'negative.npy')
    with pytest.raises(ValueError, match='negative'):
        nm.load_survey_footprint(cfg).raw_mask('wide')


def test_regrade(mask_files):
    fp = nm.load_survey_footprint(_config(mask_files, nside=8))
    with pytest.warns(UserWarning, match='regrading'):
        mask = fp.raw_mask('wide')
    assert mask.size == hp.nside2npix(8)


def test_arrays_accepted(mask_files):
    cfg = _config(mask_files)
    cfg['surveys']['footprint']['masks']['wide'] = {'file': _cap(NSIDE, 100)}
    fp = nm.load_survey_footprint(cfg)
    assert np.array_equal(fp.raw_mask('wide'), _cap(NSIDE, 100))


def test_component_index():
    assert nm._component_index('galaxy_density_cl', 1) == 0
    assert nm._component_index('galaxy_shearDensity_cl_e', 2) == 0
    assert nm._component_index('galaxy_shearDensity_cl_b', 2) == 1
    assert nm._component_index('galaxy_shear_cl_ee', 4) == 0
    assert nm._component_index('galaxy_shear_cl_bb', 4) == 3
    assert nm._component_index('galaxy_shear_cl_eb', 4) == 1
    assert nm._component_index('galaxy_shear_cl_eb', 4, swapped=True) == 2
    with pytest.raises(ValueError, match='does not match'):
        nm._component_index('galaxy_shear_cl_ee', 1)


# ---- NaMaster ------------------------------------------------------------------

@needs_pymaster
def test_make_nmt_bin():
    bins = nm.make_nmt_bin(EDGES, LMAX)
    assert bins.get_n_bands() == len(EDGES) - 1
    assert bins.lmax == LMAX
    assert list(bins.get_ell_list(0)) == list(range(2, 10))
    assert list(bins.get_ell_list(3)) == list(range(30, 48))
    with pytest.raises(ValueError, match='above the NaMaster lmax'):
        nm.make_nmt_bin([2, 10, LMAX + 2], LMAX)
    with pytest.raises(ValueError, match='no integer multipole'):
        nm.make_nmt_bin([2, 10.2, 10.8, 20], LMAX)


@needs_pymaster
def test_small_apodization_raises_instead_of_crashing(mask_files):
    cfg = _config(mask_files)
    cfg['surveys']['footprint']['masks']['wide']['apodize'] = {'scale_deg': 4.0, 'type': 'C1'}
    fp = nm.load_survey_footprint(cfg)
    with pytest.raises(ValueError, match='pymaster crashes'):
        fp.mask('wide')


@needs_pymaster
def test_apodization_applied(mask_files):
    cfg = _config(mask_files)
    cfg['surveys']['footprint']['masks']['wide']['apodize'] = {'scale_deg': 15.0, 'type': 'C1'}
    fp = nm.load_survey_footprint(cfg)
    assert fp.mask('wide').sum() < fp.raw_mask('wide').sum()
    assert np.array_equal(fp.raw_mask('wide'), _cap(NSIDE, 100))


@needs_pymaster
def test_windows_match_direct_pymaster(mask_files):
    import pymaster as nmt
    nw = nm.load_namaster(_config(mask_files, namaster={}))
    assert nw.source == 'masks'
    assert nw.lmax == LMAX
    ell_eff, win = nw.bandpower_window('lens1', 'cmb_convergence', EDGES,
                                       'cmbGalaxy_convergenceDensity_cl')
    assert isinstance(win, sacc.BandpowerWindow)

    small = _cap(NSIDE, 60)
    bins = nm.make_nmt_bin(EDGES, LMAX)
    ref = nmt.NmtWorkspace()
    ref.compute_coupling_matrix(nmt.NmtField(small, None, spin=0),
                                nmt.NmtField(small, None, spin=0), bins)
    assert np.allclose(win.weight, ref.get_bandpower_windows()[0, :, 0, :].T)
    assert np.allclose(win.values, np.arange(LMAX + 1))
    assert np.allclose(ell_eff, bins.get_effective_ells())


@needs_pymaster
def test_workspace_shared_and_rebinned(mask_files):
    nw = nm.load_namaster(_config(mask_files, namaster={}))
    assert nw.workspace('src0', 'src1') is nw.workspace('src2', 'src2')
    assert nw.workspace('src0', 'lens0') is nw.workspace('lens2', 'src1')
    assert nw.workspace('lens0', 'lens0') is not nw.workspace('lens1', 'lens1')

    coarse = [2, 20, 48]
    _, fine_win = nw.bandpower_window('src0', 'src1', EDGES, 'galaxy_shear_cl_ee')
    _, coarse_win = nw.bandpower_window('src0', 'src1', coarse, 'galaxy_shear_cl_ee')
    fresh = nm.load_namaster(_config(mask_files, namaster={}))
    _, fresh_win = fresh.bandpower_window('src0', 'src1', coarse, 'galaxy_shear_cl_ee')
    assert coarse_win.weight.shape == (LMAX + 1, 2)
    assert np.allclose(coarse_win.weight, fresh_win.weight)
    assert nw.origin[nw.tracer_pair_key('src0', 'src1')] == 'computed'


@needs_pymaster
def test_cache_roundtrip(tmp_path, mask_files):
    cache = tmp_path / 'cache'
    cfg = _config(mask_files, namaster={'cache_dir': str(cache)})
    first = nm.load_namaster(cfg)
    _, win1 = first.bandpower_window('src0', 'lens0', EDGES, 'galaxy_shearDensity_cl_e')
    first.covariance_workspace('src0', 'lens0', 'src0', 'lens0')
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert sorted(v['kind'] for v in manifest.values()) == ['covariance', 'workspace']

    second = nm.load_namaster(cfg)
    _, win2 = second.bandpower_window('src0', 'lens0', EDGES, 'galaxy_shearDensity_cl_e')
    second.covariance_workspace('lens0', 'src0', 'lens0', 'src0')
    assert set(second.summary()['workspaces'].values()) == {'cache'}
    assert set(second.summary()['covariance_workspaces'].values()) == {'cache'}
    assert np.allclose(win1.weight, win2.weight)

    cfg['surveys']['footprint']['namaster']['recompute'] = True
    third = nm.load_namaster(cfg)
    third.bandpower_window('src0', 'lens0', EDGES, 'galaxy_shearDensity_cl_e')
    assert set(third.summary()['workspaces'].values()) == {'computed'}


@needs_pymaster
def test_covariance_workspace_symmetry(mask_files):
    nw = nm.load_namaster(_config(mask_files, namaster={}))
    cw = nw.covariance_workspace('lens0', 'lens1', 'src0', 'src1')
    assert nw.covariance_workspace('src0', 'src1', 'lens1', 'lens0') is cw
    assert nw.covariance_workspace('lens0', 'lens0', 'src0', 'src1') is not cw


@needs_pymaster
def test_supplied_covariance_workspace_spin_header(tmp_path, mask_files):
    import pymaster as nmt
    from astropy.io import fits
    field = nmt.NmtField(_cap(NSIDE, 100), None, spin=0)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(field, field, field, field)
    path = tmp_path / 'cw.fits'
    cw.write_to(str(path))
    entry = {'masks': ['wide'] * 4, 'file': str(path)}
    nw = nm.load_namaster(_config(mask_files, namaster={'covariance_workspaces': [entry]}))
    nw.covariance_workspace('src0', 'lens0', 'src1', 'lens0')
    assert set(nw.cw_origin.values()) == {'provided'}

    # pymaster 3 marks files built for a single spin combination with ALL_SPINS = 0.
    fits.setval(str(path), 'ALL_SPINS', value=0, ext=0)
    with pytest.raises(ValueError, match='all_spins=True'):
        nm.load_namaster(_config(mask_files, namaster={'covariance_workspaces': [entry]}))


@needs_pymaster
def test_workspaces_mode(tmp_path, mask_files):
    path = tmp_path / 'w22.fits'
    _write_workspace(path, _cap(NSIDE, 100), 2, _cap(NSIDE, 100), 2)
    cfg = _config(mask_files, namaster={
        'source': 'workspaces',
        'workspaces': [{'masks': ['wide', 'wide'], 'file': str(path)}],
    })
    nw = nm.load_namaster(cfg)
    assert nw.lmax == LMAX
    _, win = nw.bandpower_window('src0', 'src3', [2, 20, 48], 'galaxy_shear_cl_bb')
    assert win.weight.shape == (LMAX + 1, 2)
    assert nw.origin[nw.tracer_pair_key('src0', 'src3')] == 'provided'
    with pytest.raises(ValueError, match='No workspace supplied'):
        nw.workspace('lens0', 'lens0')


@needs_pymaster
def test_mixed_mode(tmp_path, mask_files):
    path = tmp_path / 'w00.fits'
    _write_workspace(path, _cap(NSIDE, 100), 0, _cap(NSIDE, 100), 0)
    cfg = _config(mask_files, namaster={
        'workspaces': [{'masks': ['wide', 'wide'], 'spins': [0, 0], 'file': str(path)}],
    })
    nw = nm.load_namaster(cfg)
    assert nw.source == 'mixed'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nw.workspace('lens0', 'lens0')
    with pytest.warns(UserWarning, match='was not supplied; computed'):
        nw.workspace('lens1', 'lens1')
    assert nw.summary()['workspaces'] == {'wide:0 x wide:0': 'provided',
                                          'small:0 x small:0': 'computed'}


@needs_pymaster
def test_supplied_workspace_checked_against_masks(tmp_path, mask_files):
    path = tmp_path / 'w00_wide.fits'
    _write_workspace(path, _cap(NSIDE, 100), 0, _cap(NSIDE, 100), 0)
    namaster = {'workspaces': [{'masks': ['small', 'small'], 'file': str(path)}]}
    with pytest.raises(ValueError, match='does not match those masks'):
        nm.load_namaster(_config(mask_files, namaster=namaster))
    namaster['check_workspaces'] = False
    nm.load_namaster(_config(mask_files, namaster=namaster))

    # A slightly smaller footprint (about 3% less overlap) is still caught for spin 0.
    near = tmp_path / 'w00_near.fits'
    _write_workspace(near, _cap(NSIDE, 98), 0, _cap(NSIDE, 98), 0)
    namaster = {'workspaces': [{'masks': ['wide', 'wide'], 'file': str(near)}]}
    with pytest.raises(ValueError, match='does not match those masks'):
        nm.load_namaster(_config(mask_files, namaster=namaster))


@needs_pymaster
def test_spins_needed_for_mixed_spin_cross_masks(tmp_path, mask_files):
    path = tmp_path / 'w02.fits'
    _write_workspace(path, _cap(NSIDE, 60), 0, _cap(NSIDE, 100), 2)
    entry = {'masks': ['small', 'wide'], 'file': str(path)}
    with pytest.raises(ValueError, match="add 'spins'"):
        nm.load_namaster(_config(mask_files, namaster={'workspaces': [entry]}))
    entry['spins'] = [0, 0]
    with pytest.raises(ValueError, match='imply 1'):
        nm.load_namaster(_config(mask_files, namaster={'workspaces': [entry]}))
    entry['spins'] = [0, 2]
    nw = nm.load_namaster(_config(mask_files, namaster={'workspaces': [entry]}))
    assert nw.origin == {}
    nw.workspace('src0', 'lens1')
    assert nw.origin[nw.tracer_pair_key('lens1', 'src0')] == 'provided'


@needs_pymaster
@pytest.mark.parametrize('namaster, match', [
    ({'source': 'masks', 'workspaces': [{'masks': ['wide', 'wide'], 'file': 'W'}]},
     "source is 'masks'"),
    ({'source': 'workspaces'}, 'no workspace files'),
    ({'source': 'sometimes'}, 'must be one of'),
    ({'lmax': 3 * NSIDE}, 'above 3 \\* nside - 1'),
    ({'cache': 'x'}, 'Unknown key'),
])
def test_namaster_config_errors(tmp_path, mask_files, namaster, match):
    for entry in namaster.get('workspaces', []):
        path = tmp_path / 'w.fits'
        _write_workspace(path, _cap(NSIDE, 100), 2, _cap(NSIDE, 100), 2)
        entry['file'] = str(path)
    with pytest.raises(ValueError, match=match):
        nm.load_namaster(_config(mask_files, namaster=namaster))


@needs_pymaster
def test_lmax_mismatch_with_files(tmp_path, mask_files):
    path = tmp_path / 'w.fits'
    _write_workspace(path, _cap(NSIDE, 100), 2, _cap(NSIDE, 100), 2)
    cfg = _config(mask_files, namaster={
        'lmax': LMAX - 10,
        'workspaces': [{'masks': ['wide', 'wide'], 'file': str(path)}],
    })
    with pytest.raises(ValueError, match='supplied NaMaster files have lmax'):
        nm.load_namaster(cfg)


@needs_pymaster
def test_non_overlapping_masks_raise(tmp_path, mask_files):
    north = tmp_path / 'north.fits'
    south = tmp_path / 'south.fits'
    hp.write_map(str(north), _cap(NSIDE, 40), dtype=np.float64)
    hp.write_map(str(south), 1.0 - _cap(NSIDE, 140), dtype=np.float64)
    cfg = _config({'wide': str(north), 'small': str(south)}, namaster={})
    nw = nm.load_namaster(cfg)
    with pytest.raises(ValueError, match='do not overlap'):
        nw.workspace('lens0', 'lens1')


def test_module_import_does_not_need_pymaster():
    assert 'pymaster' not in nm.__dict__
    assert os.path.basename(nm.__file__) == 'namaster.py'
