"""
Survey footprints and NaMaster products.

This module reads the ``surveys`` section of an Augur config, loads the survey
masks, and either computes NaMaster mode-coupling workspaces from those masks or
loads pre-computed workspaces from file. The workspaces give the bandpower
windows written to the SACC file and can be handed to TJPCov for a NaMaster
Gaussian covariance.

pymaster is optional. It is imported only when a NaMaster product (or mask
apodization) is actually needed, so footprints work with healpy alone.

Config layout
-------------
.. code-block:: yaml

    surveys:
      footprint:
        nside: 1024                     # optional; default: nside of the first mask file
        masks:
          lsst_wfd:
            file: masks/wfd.fits
            apodize: {scale_deg: 1.0, type: C1}
          act_kappa: {file: masks/act_kappa.fits}
        namaster:
          source: masks                 # masks | workspaces | mixed (default: inferred)
          lmax: 3071                    # default: 3 * nside - 1
          cache_dir: namaster_cache     # optional; reuse computed products across runs
          workspaces:                   # pre-computed NmtWorkspace files
            - {masks: [lsst_wfd, lsst_wfd], spins: [0, 2], file: ws/w02.fits}
          covariance_workspaces:        # pre-computed NmtCovarianceWorkspace files
            - {masks: [lsst_wfd, lsst_wfd, lsst_wfd, lsst_wfd], file: ws/cw.fits}
      sources: {mask: lsst_wfd}
      lenses: {mask: lsst_wfd, bin_masks: {4: lsst_deep}}
      cmb_lensing: {mask: act_kappa}
      cluster_counts: {mask: lsst_wfd}

Workspaces are keyed by (mask, spin) pairs, never by tracer or binning, so every
tracer pair with the same masks and spins shares one coupling matrix, and a new
binning only re-bins the stored unbinned matrix.
"""

import hashlib
import json
import logging
import os
import re
import warnings

import healpy as hp
import numpy as np
import sacc
from astropy.io import fits

logger = logging.getLogger(__name__)

PROBES = ('sources', 'lenses', 'cmb_lensing', 'cluster_counts')
BINNED_PROBES = ('sources', 'lenses')
PROBE_SPINS = {'sources': 2, 'lenses': 0, 'cmb_lensing': 0}
NAMASTER_SOURCES = ('masks', 'workspaces', 'mixed')
APODIZATION_TYPES = ('C1', 'C2', 'Smooth')

# pymaster 2.7 C1/C2 apodization kills the interpreter (stack smashing) when the
# apodization scale is about one pixel; two or more pixels has always worked.
MIN_APODIZATION_PIXELS = 2.0

# Rows of an unbinned coupling matrix sum to the mask overlap
# sum_L (2L + 1) C_L(m_a, m_b) / 4 pi, band-limited to lmax. For spin 0 x 0, row
# ell = 0 matches it to machine precision. Spin-2 matrices have empty low rows, so
# their mid-ell rows are used instead; those agree only to a few per cent at
# nside <= 32 (under 1% from nside 64), so that check catches gross mismatches only.
ROW_SUM_TOL_SPIN0 = 1e-3
ROW_SUM_TOL_SPIN2 = 5e-2

_TRACER_RE = re.compile(r'^(src|lens)(\d+)$')
_TRACER_PROBES = {'src': 'sources', 'lens': 'lenses'}
_FOOTPRINT_KEYS = {'nside', 'masks', 'namaster'}
_MASK_KEYS = {'file', 'field', 'apodize'}
_APODIZE_KEYS = {'scale_deg', 'type'}
_PROBE_KEYS = {'mask', 'bin_masks'}
_NAMASTER_KEYS = {
    'source', 'lmax', 'cache_dir', 'recompute', 'check_workspaces',
    'field_kwargs', 'workspace_kwargs', 'covariance_kwargs',
    'workspaces', 'covariance_workspaces',
}
_N_SPIN_COMPONENTS = {0: 1, 2: 2}
_MANIFEST = 'manifest.json'


def import_pymaster():
    """
    Import pymaster, with an actionable error if it is missing.

    Returns
    -------
    module
        The ``pymaster`` module.
    """
    try:
        import pymaster as nmt
    except ImportError as err:
        raise ImportError(
            "NaMaster products need pymaster (package 'namaster', version >=2.7,<3). "
            "TJPCov does not support pymaster 3 yet."
        ) from err
    try:
        major = int(str(nmt.__version__).split('.')[0])
    except (AttributeError, ValueError):
        major = None
    if major is not None and major >= 3:
        warnings.warn(
            f"pymaster {nmt.__version__} is untested with Augur; TJPCov requires pymaster <3."
        )
    return nmt


def tracer_probe(tracer_name):
    """
    Map a SACC tracer name to its ``surveys`` probe key and bin index.

    Parameters
    ----------
    tracer_name : str
        SACC tracer name, e.g. ``'src2'``, ``'lens0'`` or ``'cmb_convergence'``.

    Returns
    -------
    probe : str
        One of ``'sources'``, ``'lenses'`` or ``'cmb_lensing'``.
    bin_index : int or None
        Tomographic bin index, or None for un-binned tracers.
    """
    if tracer_name == 'cmb_convergence':
        return 'cmb_lensing', None
    match = _TRACER_RE.match(str(tracer_name))
    if match is None:
        raise ValueError(
            f"Tracer '{tracer_name}' has no NaMaster field. Expected src<i>, lens<i> "
            "or cmb_convergence."
        )
    return _TRACER_PROBES[match.group(1)], int(match.group(2))


def make_nmt_bin(ell_edges, lmax):
    """
    Build an NmtBin whose band ``i`` holds the integers ``edge_i <= ell < edge_{i+1}``.

    The binning is built with an explicit ``lmax`` because pymaster requires the
    bins and the fields to share it (``NmtBin.from_edges`` would set it to the
    last edge minus one).

    Parameters
    ----------
    ell_edges : array_like
        Increasing band edges. The last edge may be at most ``lmax + 1``.
    lmax : int
        Maximum multipole of the NaMaster fields.

    Returns
    -------
    pymaster.NmtBin
    """
    nmt = import_pymaster()
    edges = np.asarray(ell_edges, dtype=float).ravel()
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("ell_edges must be increasing and have at least two entries.")
    if edges[0] < 0:
        raise ValueError(f"ell_edges must be non-negative; the first edge is {edges[0]:g}.")
    if edges[-1] > lmax + 1:
        raise ValueError(
            f"The last ell edge ({edges[-1]:g}) is above the NaMaster lmax ({lmax}) + 1. "
            "Lower ell_edges, or raise surveys.footprint.nside or namaster.lmax."
        )
    ells = np.arange(lmax + 1)
    band = np.searchsorted(edges, ells, side='right') - 1
    keep = (band >= 0) & (band < edges.size - 1)
    counts = np.bincount(band[keep], minlength=edges.size - 1)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        bounds = ', '.join(f"[{edges[i]:g}, {edges[i + 1]:g})" for i in empty)
        raise ValueError(
            f"ell bands {empty.tolist()} contain no integer multipole: {bounds}. "
            "Use wider or integer edges."
        )
    return nmt.NmtBin(bpws=band[keep], ells=ells[keep],
                      weights=np.ones(int(keep.sum())), lmax=lmax)


def _band_limited_overlap(mask_a, mask_b, lmax):
    """sum_{L <= lmax} (2L + 1) C_L(m_a, m_b) / 4 pi, i.e. mean(m_a m_b) without L > lmax."""
    cl = hp.anafast(mask_a, mask_b, lmax=lmax, iter=3)
    return float(np.sum((2 * np.arange(lmax + 1) + 1) * cl) / (4 * np.pi))


def _hash_array(arr):
    digest = hashlib.sha1(str(arr.size).encode())
    digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def _hash_settings(settings):
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _check_keys(cfg, allowed, where):
    unknown = set(cfg) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {where}. Allowed: {sorted(allowed)}."
        )


def _parse_mask_spec(name, spec):
    where = f"surveys.footprint.masks.{name}"
    if spec is None:
        spec = {}
    elif isinstance(spec, (str, os.PathLike, np.ndarray)):
        spec = {'file': spec}
    elif not isinstance(spec, dict):
        raise TypeError(f"{where} must be a file path or a mapping, got {type(spec).__name__}.")
    _check_keys(spec, _MASK_KEYS, where)

    apodize = spec.get('apodize')
    if apodize is not None:
        if not isinstance(apodize, dict) or 'scale_deg' not in apodize:
            raise ValueError(f"{where}.apodize needs at least 'scale_deg'.")
        _check_keys(apodize, _APODIZE_KEYS, f"{where}.apodize")
        apotype = apodize.get('type', 'C1')
        if apotype not in APODIZATION_TYPES:
            raise ValueError(
                f"{where}.apodize.type must be one of {APODIZATION_TYPES}, got '{apotype}'."
            )
        scale = float(apodize['scale_deg'])
        if scale <= 0:
            raise ValueError(f"{where}.apodize.scale_deg must be positive, got {scale:g}.")
        apodize = {'scale_deg': scale, 'type': apotype}
    return {'file': spec.get('file'), 'field': int(spec.get('field', 0)), 'apodize': apodize}


def _read_mask(name, source, field):
    if isinstance(source, np.ndarray):
        arr = np.array(source, dtype=np.float64)
    else:
        path = os.fspath(source)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mask '{name}': file not found: {path}")
        if path.endswith('.npy'):
            arr = np.load(path)
        else:
            arr = hp.read_map(path, field=field, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 1 or not hp.isnpixok(arr.size):
        raise ValueError(f"Mask '{name}' is not a HEALPix map (shape {arr.shape}).")
    bad = ~np.isfinite(arr) | (arr == hp.UNSEEN)
    if np.any(bad):
        warnings.warn(
            f"Mask '{name}': {int(bad.sum())} NaN, inf or UNSEEN pixels set to 0."
        )
        arr = np.where(bad, 0.0, arr)
    if np.any(arr < 0):
        raise ValueError(f"Mask '{name}' has negative pixels (min {arr.min():g}).")
    if not np.any(arr > 0):
        raise ValueError(f"Mask '{name}' is zero everywhere.")
    return arr


class SurveyFootprint:
    """
    Survey masks and the probe-to-mask assignment from ``config['surveys']``.

    Masks are read, cleaned, regraded to a common ``nside`` and apodized the
    first time they are needed, then kept in memory. Only apodization needs
    pymaster.

    Parameters
    ----------
    surveys_cfg : dict
        The ``surveys`` section of an Augur config.
    """

    def __init__(self, surveys_cfg):
        if not isinstance(surveys_cfg, dict):
            raise TypeError("The 'surveys' config section must be a mapping.")
        _check_keys(surveys_cfg, {'footprint', *PROBES}, 'surveys')
        fp_cfg = surveys_cfg.get('footprint')
        if not isinstance(fp_cfg, dict):
            raise ValueError("'surveys' needs a 'footprint' mapping.")
        _check_keys(fp_cfg, _FOOTPRINT_KEYS, 'surveys.footprint')

        self.namaster_enabled = 'namaster' in fp_cfg
        self.namaster_cfg = fp_cfg.get('namaster') or {}

        masks_cfg = fp_cfg.get('masks') or {}
        if not isinstance(masks_cfg, dict):
            raise TypeError("surveys.footprint.masks must map mask names to specs.")
        self._specs = {str(n): _parse_mask_spec(n, s) for n, s in masks_cfg.items()}

        self._nside = fp_cfg.get('nside')
        if self._nside is not None:
            self._nside = int(self._nside)
            if not hp.isnsideok(self._nside):
                raise ValueError(f"surveys.footprint.nside = {self._nside} is not a valid nside.")

        self._probes = {p: self._parse_probe(p, surveys_cfg[p])
                        for p in PROBES if p in surveys_cfg}
        self._native = {}
        self._raw = {}
        self._masks = {}
        self._hashes = {}

    # ---- config ------------------------------------------------------------

    def _check_mask_name(self, name, where):
        if name not in self._specs:
            raise ValueError(
                f"{where} names mask '{name}', which is not defined in "
                f"surveys.footprint.masks ({sorted(self._specs)})."
            )

    def _parse_probe(self, probe, cfg):
        where = f"surveys.{probe}"
        if isinstance(cfg, str):
            cfg = {'mask': cfg}
        if not isinstance(cfg, dict) or 'mask' not in cfg:
            raise ValueError(f"{where} needs a 'mask' entry.")
        _check_keys(cfg, _PROBE_KEYS, where)
        mask = str(cfg['mask'])
        self._check_mask_name(mask, where)

        bin_masks = cfg.get('bin_masks') or {}
        if bin_masks and probe not in BINNED_PROBES:
            raise ValueError(f"{where}.bin_masks is only allowed for {BINNED_PROBES}.")
        parsed = {}
        for key, name in bin_masks.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                raise ValueError(f"{where}.bin_masks keys must be bin indices, got '{key}'.")
            if index < 0:
                raise ValueError(f"{where}.bin_masks has a negative bin index {index}.")
            self._check_mask_name(str(name), f"{where}.bin_masks.{index}")
            parsed[index] = str(name)
        return {'mask': mask, 'bin_masks': parsed}

    @property
    def mask_names(self):
        """Names of all masks defined in ``surveys.footprint.masks``."""
        return list(self._specs)

    @property
    def probes(self):
        """Probes with an entry under ``surveys``."""
        return list(self._probes)

    def has_mask_file(self, name):
        """True if mask ``name`` has a file (or array) to load."""
        self._check_mask_name(name, 'has_mask_file')
        return self._specs[name]['file'] is not None

    def probe_mask_name(self, probe, bin_index=None):
        """Mask name for ``probe``, honouring per-bin overrides."""
        if probe not in self._probes:
            raise ValueError(
                f"'surveys' has no '{probe}' entry; add one naming its mask, "
                f"e.g. {probe}: {{mask: <name>}}."
            )
        entry = self._probes[probe]
        return entry['bin_masks'].get(bin_index, entry['mask'])

    def tracer_mask_name(self, tracer_name):
        """Mask name for a SACC tracer such as ``'src0'`` or ``'cmb_convergence'``."""
        return self.probe_mask_name(*tracer_probe(tracer_name))

    @staticmethod
    def tracer_spin(tracer_name):
        """NaMaster field spin of a SACC tracer (2 for shear, 0 otherwise)."""
        return PROBE_SPINS[tracer_probe(tracer_name)[0]]

    # ---- masks -------------------------------------------------------------

    def _native_mask(self, name):
        if name not in self._native:
            spec = self._specs[name]
            if spec['file'] is None:
                raise ValueError(
                    f"Mask '{name}' has no file in surveys.footprint.masks, "
                    "but this step needs the mask itself."
                )
            self._native[name] = _read_mask(name, spec['file'], spec['field'])
        return self._native[name]

    @property
    def nside(self):
        """
        Common nside: ``footprint.nside`` if set, else that of the first mask file.

        None if neither is available (for example, workspace files only).
        """
        if self._nside is None:
            for name in self._specs:
                if self._specs[name]['file'] is not None:
                    self._nside = hp.npix2nside(self._native_mask(name).size)
                    break
        return self._nside

    def raw_mask(self, name):
        """Mask ``name`` at the common nside, before apodization (use for areas)."""
        self._check_mask_name(name, 'raw_mask')
        if name not in self._raw:
            arr = self._native_mask(name)
            native = hp.npix2nside(arr.size)
            if native != self.nside:
                warnings.warn(
                    f"Mask '{name}' has nside {native}; regrading to nside {self.nside}."
                )
                arr = hp.ud_grade(arr, self.nside)
            self._raw[name] = arr
        return self._raw[name]

    def mask(self, name):
        """Mask ``name`` as NaMaster uses it: regraded and, if configured, apodized."""
        self._check_mask_name(name, 'mask')
        if name not in self._masks:
            raw = self.raw_mask(name)
            apodize = self._specs[name]['apodize']
            if apodize is None:
                self._masks[name] = raw
            else:
                pixel_deg = np.degrees(hp.nside2resol(self.nside))
                if apodize['scale_deg'] < MIN_APODIZATION_PIXELS * pixel_deg:
                    raise ValueError(
                        f"Mask '{name}': apodization scale {apodize['scale_deg']:g} deg is "
                        f"under {MIN_APODIZATION_PIXELS:g} pixels ({pixel_deg:.3g} deg each at "
                        f"nside {self.nside}); pymaster crashes at about one pixel. "
                        "Use a larger scale or a higher nside."
                    )
                nmt = import_pymaster()
                self._masks[name] = nmt.mask_apodization(
                    raw, apodize['scale_deg'], apotype=apodize['type']
                )
        return self._masks[name]

    def mask_hash(self, name):
        """Content hash of :meth:`mask`, used to key cached NaMaster products."""
        if name not in self._hashes:
            self._hashes[name] = _hash_array(self.mask(name))
        return self._hashes[name]

    def mean_overlap(self, name_a, name_b):
        """Mean over the sphere of ``mask(a) * mask(b)``."""
        return float(np.mean(self.mask(name_a) * self.mask(name_b)))


def _pair_key(mask_a, spin_a, mask_b, spin_b):
    first, second = (mask_a, int(spin_a)), (mask_b, int(spin_b))
    return (first, second) if first <= second else (second, first)


def _quartet_permutations(a, b, c, d):
    """The orderings TJPCov treats as the same covariance workspace."""
    return [(a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
            (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a)]


def _cw_key(*names):
    return min(_quartet_permutations(*names))


def _component_index(data_type, ncls, swapped=False):
    """Index of the spectrum within a workspace's ``ncls`` spectra for ``data_type``."""
    if data_type is None:
        return 0
    parts = str(data_type).lower().rsplit('_cl', 1)
    suffix = parts[1].lstrip('_') if len(parts) == 2 else ''
    table = {
        1: {'': 0},
        2: {'e': 0, 'b': 1},
        4: {'ee': 0, 'eb': 1, 'be': 2, 'bb': 3},
    }[ncls]
    if swapped and ncls == 4:
        table = {'ee': 0, 'eb': 2, 'be': 1, 'bb': 3}
    if suffix not in table:
        raise ValueError(
            f"Data type '{data_type}' does not match a workspace with {ncls} spectra."
        )
    return table[suffix]


class NaMasterWorkspaces:
    """
    NaMaster workspaces and covariance workspaces for a :class:`SurveyFootprint`.

    Depending on ``namaster.source``, products are computed from the masks
    (``masks``), read from the listed files only (``workspaces``), or read where
    listed and computed otherwise (``mixed``). If ``source`` is omitted it is
    inferred from what the config provides.

    Parameters
    ----------
    footprint : SurveyFootprint
        Survey masks and probe assignments.
    namaster_cfg : dict, optional
        NaMaster settings; defaults to ``surveys.footprint.namaster``.
    """

    def __init__(self, footprint, namaster_cfg=None):
        cfg = footprint.namaster_cfg if namaster_cfg is None else (namaster_cfg or {})
        if not isinstance(cfg, dict):
            raise TypeError("surveys.footprint.namaster must be a mapping.")
        _check_keys(cfg, _NAMASTER_KEYS, 'surveys.footprint.namaster')

        self.footprint = footprint
        self._nmt = import_pymaster()
        self.cache_dir = cfg.get('cache_dir')
        self.recompute = bool(cfg.get('recompute', False))
        self.check_workspaces = bool(cfg.get('check_workspaces', True))
        self.field_kwargs = dict(cfg.get('field_kwargs') or {})
        self.workspace_kwargs = dict(cfg.get('workspace_kwargs') or {})
        self.covariance_kwargs = dict(cfg.get('covariance_kwargs') or {})

        self._workspaces = {}
        self._edges = {}
        self._fields = {}
        self._cws = {}
        self.origin = {}
        self.cw_origin = {}

        self._provided = {}
        for i, entry in enumerate(cfg.get('workspaces') or []):
            key, wsp = self._load_provided_workspace(entry, f"namaster.workspaces[{i}]")
            if key in self._provided:
                raise ValueError(f"namaster.workspaces lists {key} more than once.")
            self._provided[key] = wsp
        self._provided_cw = {}
        for i, entry in enumerate(cfg.get('covariance_workspaces') or []):
            key, cwsp = self._load_provided_cw(entry, f"namaster.covariance_workspaces[{i}]")
            if key in self._provided_cw:
                raise ValueError(f"namaster.covariance_workspaces lists {key} more than once.")
            self._provided_cw[key] = cwsp

        self.source = self._resolve_source(cfg.get('source'))
        self.lmax = self._resolve_lmax(cfg.get('lmax'))
        if self.check_workspaces:
            for key, wsp in self._provided.items():
                self._check_against_masks(key, wsp)

    # ---- setup -------------------------------------------------------------

    def _read_product(self, source, cls, where):
        if isinstance(source, cls):
            return source
        path = os.fspath(source)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{where}: file not found: {path}")
        product = cls()
        product.read_from(path)
        return product

    def _entry_masks(self, entry, count, where):
        if not isinstance(entry, dict) or 'masks' not in entry or 'file' not in entry:
            raise ValueError(f"{where} needs 'masks' and 'file'.")
        masks = [str(m) for m in entry['masks']]
        if len(masks) != count:
            raise ValueError(f"{where}.masks must list {count} mask names, got {masks}.")
        for name in masks:
            self.footprint._check_mask_name(name, where)
        return masks

    def _load_provided_workspace(self, entry, where):
        masks = self._entry_masks(entry, 2, where)
        _check_keys(entry, {'masks', 'spins', 'file'}, where)
        wsp = self._read_product(entry['file'], self._nmt.NmtWorkspace, where)
        ncls = int(wsp.wsp.ncls)
        spins = entry.get('spins')
        if spins is None:
            if ncls == 1:
                spins = [0, 0]
            elif ncls == 4:
                spins = [2, 2]
            elif masks[0] == masks[1]:
                spins = [0, 2]
            else:
                raise ValueError(
                    f"{where} couples a spin-0 and a spin-2 field on different masks; "
                    "add 'spins' to say which mask carries which."
                )
        spins = [int(s) for s in spins]
        if len(spins) != 2 or any(s not in _N_SPIN_COMPONENTS for s in spins):
            raise ValueError(f"{where}.spins must be two values from (0, 2), got {spins}.")
        expected = _N_SPIN_COMPONENTS[spins[0]] * _N_SPIN_COMPONENTS[spins[1]]
        if ncls != expected:
            raise ValueError(
                f"{where} holds {ncls} spectra, but spins {spins} imply {expected}."
            )
        return _pair_key(masks[0], spins[0], masks[1], spins[1]), wsp

    def _load_provided_cw(self, entry, where):
        masks = self._entry_masks(entry, 4, where)
        _check_keys(entry, {'masks', 'file'}, where)
        source = entry['file']
        if not isinstance(source, self._nmt.NmtCovarianceWorkspace) and os.path.isfile(source):
            # pymaster 3 writes ALL_SPINS = 0 for files that only cover the spins they were
            # built with; pymaster 2 cannot read those (and its files have no such key).
            if fits.getheader(source, 0).get('ALL_SPINS', 1) == 0:
                raise ValueError(
                    f"{where}: {source} was written by pymaster 3 for one spin combination "
                    "only, which pymaster 2 cannot use. Regenerate it with "
                    "NmtCovarianceWorkspace.from_fields(..., all_spins=True)."
                )
        cwsp = self._read_product(source, self._nmt.NmtCovarianceWorkspace, where)
        return _cw_key(*masks), cwsp

    def _resolve_source(self, source):
        has_products = bool(self._provided or self._provided_cw)
        has_files = any(self.footprint.has_mask_file(n) for n in self.footprint.mask_names)
        if source is None:
            if has_products and has_files:
                source = 'mixed'
            elif has_products:
                source = 'workspaces'
            else:
                source = 'masks'
        if source not in NAMASTER_SOURCES:
            raise ValueError(f"namaster.source must be one of {NAMASTER_SOURCES}, got '{source}'.")
        if source == 'masks' and has_products:
            raise ValueError(
                "namaster.source is 'masks' but workspace files are listed; remove them "
                "or set source: mixed."
            )
        if source == 'workspaces' and not has_products:
            raise ValueError("namaster.source is 'workspaces' but no workspace files are listed.")
        if source in ('masks', 'mixed') and not has_files:
            raise ValueError(
                f"namaster.source is '{source}' but no mask in surveys.footprint.masks "
                "has a file."
            )
        return source

    def _resolve_lmax(self, lmax):
        file_lmax = {int(w.wsp.lmax) for w in self._provided.values()}
        file_lmax |= {int(c.wsp.lmax) for c in self._provided_cw.values()}
        if len(file_lmax) > 1:
            raise ValueError(
                f"The supplied NaMaster files have different lmax: {sorted(file_lmax)}."
            )
        nside = self.footprint.nside
        if lmax is not None:
            lmax = int(lmax)
        elif file_lmax:
            lmax = next(iter(file_lmax))
        elif nside is not None:
            lmax = 3 * nside - 1
        else:
            raise ValueError("Set namaster.lmax or surveys.footprint.nside, or give mask files.")
        if file_lmax and lmax not in file_lmax:
            raise ValueError(
                f"namaster.lmax = {lmax}, but the supplied NaMaster files have lmax "
                f"{next(iter(file_lmax))}."
            )
        if nside is not None and lmax > 3 * nside - 1:
            raise ValueError(
                f"lmax = {lmax} is above 3 * nside - 1 = {3 * nside - 1} for nside {nside}."
            )
        return lmax

    def _check_against_masks(self, key, wsp):
        """Compare a supplied workspace's row sums with the configured masks."""
        (mask_a, spin_a), (mask_b, spin_b) = key
        fp = self.footprint
        if not (fp.has_mask_file(mask_a) and fp.has_mask_file(mask_b)):
            return
        try:
            wsp.check_unbinned()
        except (RuntimeError, ValueError):
            warnings.warn(f"Workspace {key} has no unbinned coupling matrix; skipping its check.")
            return
        lmax = int(wsp.wsp.lmax)
        rows = wsp.get_coupling_matrix().sum(axis=1).reshape(lmax + 1, -1)
        expected = _band_limited_overlap(fp.mask(mask_a), fp.mask(mask_b),
                                         min(lmax, int(wsp.wsp.lmax_mask)))
        if spin_a == spin_b == 0:
            measured, tol = float(rows[0, 0]), ROW_SUM_TOL_SPIN0
        else:
            measured = float(np.median(rows[lmax // 4: lmax // 2 + 1]))
            tol = ROW_SUM_TOL_SPIN2
        rel = abs(measured / expected - 1)
        if rel > tol:
            raise ValueError(
                f"The workspace supplied for {mask_a} (spin {spin_a}) x {mask_b} (spin {spin_b}) "
                f"does not match those masks: its coupling matrix implies a mask overlap "
                f"of {measured:.6g}, but the masks give {expected:.6g} ({rel:.2%} apart, "
                f"tolerance {tol:.1%}). Check the files, or set check_workspaces: false."
            )

    # ---- cache -------------------------------------------------------------

    def _cache_path(self, prefix, settings):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"{prefix}_{_hash_settings(settings)}.fits")

    def _record(self, path, info):
        os.makedirs(self.cache_dir, exist_ok=True)
        manifest_path = os.path.join(self.cache_dir, _MANIFEST)
        manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path) as handle:
                manifest = json.load(handle)
        info = dict(info, nside=self.footprint.nside, lmax=self.lmax,
                    pymaster_version=str(getattr(self._nmt, '__version__', 'unknown')))
        manifest[os.path.basename(path)] = info
        tmp = manifest_path + '.tmp'
        with open(tmp, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp, manifest_path)

    def _settings(self, kind, ids, kwargs):
        return {'kind': kind, 'ids': ids, 'lmax': self.lmax,
                'field_kwargs': self.field_kwargs, 'kwargs': kwargs}

    # ---- workspaces ----------------------------------------------------------

    def _field(self, mask_name, spin):
        key = (mask_name, spin)
        if key not in self._fields:
            self._fields[key] = self._nmt.NmtField(
                self.footprint.mask(mask_name), None, spin=spin, lmax=self.lmax,
                **self.field_kwargs
            )
        return self._fields[key]

    def _default_bins(self):
        edges = np.unique(np.linspace(2, self.lmax + 1, min(21, self.lmax)).astype(int))
        return make_nmt_bin(edges, self.lmax)

    def _compute_or_load(self, key, bins):
        (mask_a, spin_a), (mask_b, spin_b) = key
        fp = self.footprint
        for name in (mask_a, mask_b):
            if not fp.has_mask_file(name):
                raise ValueError(
                    f"No workspace supplied for {mask_a} (spin {spin_a}) x {mask_b} "
                    f"(spin {spin_b}), and mask '{name}' has no file to compute it from."
                )
        if fp.mean_overlap(mask_a, mask_b) <= 0:
            raise ValueError(
                f"Masks '{mask_a}' and '{mask_b}' do not overlap, so their cross-spectra "
                "cannot be measured."
            )
        ids = sorted([[fp.mask_hash(mask_a), spin_a], [fp.mask_hash(mask_b), spin_b]])
        path = self._cache_path(f"w{ids[0][1]}{ids[1][1]}",
                                self._settings('workspace', ids, self.workspace_kwargs))
        wsp = self._nmt.NmtWorkspace()
        if path is not None and os.path.isfile(path) and not self.recompute:
            wsp.read_from(path)
            wsp.update_bins(bins)
            return wsp, 'cache'

        logger.info("Computing NaMaster workspace %s (spin %d) x %s (spin %d), lmax %d",
                    mask_a, spin_a, mask_b, spin_b, self.lmax)
        wsp.compute_coupling_matrix(self._field(mask_a, spin_a), self._field(mask_b, spin_b),
                                    bins, **self.workspace_kwargs)
        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            wsp.write_to(path)
            self._record(path, {'kind': 'workspace', 'masks': [mask_a, mask_b],
                                'spins': [spin_a, spin_b],
                                'mask_hashes': [fp.mask_hash(mask_a), fp.mask_hash(mask_b)],
                                'kwargs': self.workspace_kwargs})
        return wsp, 'computed'

    def _workspace(self, key, bins=None, edges=None):
        if key not in self._workspaces:
            if key in self._provided:
                wsp, origin = self._provided[key], 'provided'
                if bins is not None:
                    wsp.update_bins(bins)
            elif self.source == 'workspaces':
                raise ValueError(
                    f"No workspace supplied for {key[0][0]} (spin {key[0][1]}) x {key[1][0]} "
                    f"(spin {key[1][1]}). Add it to surveys.footprint.namaster.workspaces, "
                    "or set source: mixed to compute missing ones from the masks."
                )
            else:
                wsp, origin = self._compute_or_load(
                    key, bins if bins is not None else self._default_bins()
                )
                if self.source == 'mixed':
                    warnings.warn(
                        f"NaMaster workspace {key[0][0]} (spin {key[0][1]}) x {key[1][0]} "
                        f"(spin {key[1][1]}) was not supplied; {origin} from the masks."
                    )
            self._workspaces[key] = wsp
            self._edges[key] = None if edges is None else np.asarray(edges, dtype=float)
            self.origin[key] = origin
        elif bins is not None:
            current = self._edges[key]
            new = np.asarray(edges, dtype=float)
            if current is None or current.shape != new.shape or not np.array_equal(current, new):
                self._workspaces[key].update_bins(bins)
                self._edges[key] = new
        return self._workspaces[key]

    def tracer_pair_key(self, tracer1, tracer2):
        """(mask, spin) key of the workspace shared by a tracer pair."""
        fp = self.footprint
        return _pair_key(fp.tracer_mask_name(tracer1), fp.tracer_spin(tracer1),
                         fp.tracer_mask_name(tracer2), fp.tracer_spin(tracer2))

    def workspace(self, tracer1, tracer2, ell_edges=None):
        """
        NmtWorkspace for a tracer pair.

        Parameters
        ----------
        tracer1, tracer2 : str
            SACC tracer names.
        ell_edges : array_like, optional
            Re-bin the workspace to these edges. Note that the workspace object is
            shared by every tracer pair with the same masks and spins, so this
            changes its binning for all of them.

        Returns
        -------
        pymaster.NmtWorkspace
        """
        key = self.tracer_pair_key(tracer1, tracer2)
        bins = None if ell_edges is None else make_nmt_bin(ell_edges, self.lmax)
        return self._workspace(key, bins, ell_edges)

    def bandpower_window(self, tracer1, tracer2, ell_edges, data_type=None):
        """
        Effective multipoles and SACC bandpower window for one tracer pair.

        Parameters
        ----------
        tracer1, tracer2 : str
            SACC tracer names, in the order of the data type.
        ell_edges : array_like
            Band edges; band ``i`` holds integers ``edge_i <= ell < edge_{i+1}``.
        data_type : str, optional
            SACC data type, used to pick the E or B component for spin-2 tracers.

        Returns
        -------
        ell_eff : numpy.ndarray
            Mean multipole of each band (``NmtBin.get_effective_ells``).
        window : sacc.BandpowerWindow
            Windows on ``ell = 0 .. lmax``.
        """
        key = self.tracer_pair_key(tracer1, tracer2)
        bins = make_nmt_bin(ell_edges, self.lmax)
        windows = self._workspace(key, bins, ell_edges).get_bandpower_windows()
        fp = self.footprint
        first = (fp.tracer_mask_name(tracer1), fp.tracer_spin(tracer1))
        icl = _component_index(data_type, windows.shape[0], swapped=first != key[0])
        weights = windows[icl, :, icl, :].T
        return bins.get_effective_ells(), sacc.BandpowerWindow(np.arange(weights.shape[0]),
                                                               weights)

    # ---- covariance workspaces ---------------------------------------------

    def covariance_workspace(self, tracer1, tracer2, tracer3, tracer4):
        """NmtCovarianceWorkspace for the covariance of ``C(1,2)`` with ``C(3,4)``."""
        fp = self.footprint
        names = [fp.tracer_mask_name(t) for t in (tracer1, tracer2, tracer3, tracer4)]
        return self.covariance_workspace_from_masks(*names)

    def covariance_workspace_from_masks(self, mask1, mask2, mask3, mask4):
        """
        NmtCovarianceWorkspace for four masks.

        Covariance workspaces depend only on the masks (not on spins or binning),
        so one is shared by every block with the same four masks.
        """
        key = _cw_key(mask1, mask2, mask3, mask4)
        if key in self._cws:
            return self._cws[key]
        if key in self._provided_cw:
            cwsp, origin = self._provided_cw[key], 'provided'
        elif self.source == 'workspaces':
            raise ValueError(
                f"No covariance workspace supplied for masks {list(key)}. Add it to "
                "surveys.footprint.namaster.covariance_workspaces, or set source: mixed."
            )
        else:
            cwsp, origin = self._compute_or_load_cw(key)
            if self.source == 'mixed':
                warnings.warn(
                    f"NaMaster covariance workspace {list(key)} was not supplied; "
                    f"{origin} from the masks."
                )
        self._cws[key] = cwsp
        self.cw_origin[key] = origin
        return cwsp

    def _compute_or_load_cw(self, key):
        fp = self.footprint
        for name in set(key):
            if not fp.has_mask_file(name):
                raise ValueError(
                    f"No covariance workspace supplied for masks {list(key)}, and mask "
                    f"'{name}' has no file to compute it from."
                )
        ids = list(min(_quartet_permutations(*[fp.mask_hash(n) for n in key])))
        path = self._cache_path('cw', self._settings('covariance', ids, self.covariance_kwargs))
        cwsp = self._nmt.NmtCovarianceWorkspace()
        if path is not None and os.path.isfile(path) and not self.recompute:
            cwsp.read_from(path)
            return cwsp, 'cache'

        logger.info("Computing NaMaster covariance workspace for masks %s, lmax %d",
                    list(key), self.lmax)
        cwsp.compute_coupling_coefficients(*[self._field(n, 0) for n in key],
                                           **self.covariance_kwargs)
        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            cwsp.write_to(path)
            self._record(path, {'kind': 'covariance', 'masks': list(key), 'mask_hashes': ids,
                                'kwargs': self.covariance_kwargs})
        return cwsp, 'computed'

    # ---- reporting ---------------------------------------------------------

    def summary(self):
        """Where each NaMaster product came from, for logs and SACC metadata."""
        def label(key):
            return ' x '.join(f"{m}:{s}" for m, s in key)
        return {
            'source': self.source,
            'lmax': self.lmax,
            'nside': self.footprint.nside,
            'workspaces': {label(k): v for k, v in self.origin.items()},
            'covariance_workspaces': {' x '.join(k): v for k, v in self.cw_origin.items()},
        }


def load_survey_footprint(config):
    """
    Build a :class:`SurveyFootprint` from ``config['surveys']``.

    Returns
    -------
    SurveyFootprint or None
        None if the config has no ``surveys`` section.
    """
    surveys = config.get('surveys')
    return None if surveys is None else SurveyFootprint(surveys)


def load_namaster(config, footprint=None):
    """
    Build :class:`NaMasterWorkspaces` if ``surveys.footprint.namaster`` is present.

    Parameters
    ----------
    config : dict
        Augur config.
    footprint : SurveyFootprint, optional
        Reuse an already-built footprint (and its loaded masks).

    Returns
    -------
    NaMasterWorkspaces or None
        None if the config does not ask for NaMaster.
    """
    footprint = footprint or load_survey_footprint(config)
    if footprint is None or not footprint.namaster_enabled:
        return None
    return NaMasterWorkspaces(footprint)
