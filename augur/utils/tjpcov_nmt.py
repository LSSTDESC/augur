"""
TJPCov's NaMaster Gaussian covariance, fed by Augur's shared NaMaster products.

Kept apart from :mod:`augur.utils.cov_utils` because TJPCov's NaMaster module
imports pymaster on load.
"""
import shutil
import tempfile

import numpy as np
from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt

from augur.generate_utils.namaster import make_nmt_bin, tracer_probe
from augur.utils.cov_utils import AugurTracerInfo


def _has_field(tracer_name):
    try:
        tracer_probe(tracer_name)
    except ValueError:
        return False
    return True


class TJPCovNmt(AugurTracerInfo, FourierGaussianNmt):
    """
    TJPCov's ``FourierGaussianNmt`` using a :class:`NaMasterWorkspaces` object.

    Fields, workspaces and covariance workspaces come from ``namaster``, where they
    are computed once and shared with the bandpower windows, instead of being
    rebuilt or re-read from disk for every block. TJPCov saves each block to its
    output directory and reloads it by tracer name on later runs, so unless an
    ``outdir`` is given this class uses a temporary directory removed by
    :meth:`cleanup` (or on leaving a ``with`` block).

    Parameters
    ----------
    config : dict
        TJPCov config ``{'tjpcov': {...}}``, as built for :class:`TJPCovGaus`.
    namaster : NaMasterWorkspaces
        Shared NaMaster products. Every mask used must have a file.
    ell_edges : array_like
        Band edges shared by every harmonic statistic in the SACC.
    """

    def __init__(self, config, namaster, ell_edges):
        tj = dict(config['tjpcov'])
        footprint = namaster.footprint
        names = {t: footprint.tracer_mask_name(t)
                 for t in tj['sacc_file'].tracers if _has_field(t)}
        tj['mask_names'] = names
        tj['mask_file'] = {t: footprint.mask(n) for t, n in names.items()}
        tj['binning_info'] = make_nmt_bin(ell_edges, namaster.lmax)
        self._tmpdir = None
        if tj.get('outdir') is None:
            self._tmpdir = tj['outdir'] = tempfile.mkdtemp(prefix='augur_tjpcov_')
        super().__init__(dict(config, tjpcov=tj))
        self.namaster = namaster
        self.ell_edges = np.asarray(ell_edges, dtype=float)
        self._concise_sacc = None
        self._check_all_bands()

    def _check_all_bands(self):
        """TJPCov reshapes every block with the band count of the first pair."""
        sacc_file = self.io.get_sacc_file()
        nbands = self.ell_edges.size - 1
        for dtype in sacc_file.get_data_types():
            for comb in sacc_file.get_tracer_combinations(data_type=dtype):
                if not all(t in self.mask_names for t in comb):
                    continue
                n = len(sacc_file.indices(data_type=dtype, tracers=comb))
                if n != nbands:
                    raise ValueError(
                        f"{dtype} {comb} has {n} bandpowers in the SACC but the NaMaster "
                        f"binning has {nbands}. The NaMaster covariance needs every band "
                        "stored for every pair (general.ignore_scale_cuts: True); scale "
                        "cuts are then applied by the likelihood filters."
                    )

    def get_covariance_block(self, tracer_comb1, tracer_comb2, use_coupled_noise=False,
                             **kwargs):
        """As TJPCov, but uncoupled (white) noise by default, which is Augur's model."""
        return super().get_covariance_block(tracer_comb1, tracer_comb2,
                                            use_coupled_noise=use_coupled_noise, **kwargs)

    def get_fields_dict(self, tracer_names, cache=None, masks=None, **kwargs):
        names = self.get_mask_names_dict(tracer_names)
        spins = self.get_tracers_spin_dict(tracer_names)
        return {i: self.namaster.field(names[i], spins[i]) for i in range(1, 5)}

    def get_workspaces_dict(self, tracer_names, bins, cache=None, fields=None, masks=None,
                            **kwargs):
        names = self.get_mask_names_dict(tracer_names)
        overlap = self.namaster.footprint.mean_overlap
        workspaces = {}
        for i in (12, 34, 13, 23, 14, 24):
            a, b = divmod(i, 10)
            if i not in (12, 34) and overlap(names[a], names[b]) == 0:
                workspaces[i] = None  # TJPCov's convention for non-overlapping masks
            else:
                workspaces[i] = self.namaster.workspace(tracer_names[a], tracer_names[b],
                                                        self.ell_edges)
        return workspaces

    def get_covariance_workspace(self, f1, f2, f3, f4, m1, m2, m3, m4, **kwargs):
        return self.namaster.covariance_workspace_from_masks(m1, m2, m3, m4)

    def get_sacc_with_concise_dtypes(self):
        # TJPCov copies the whole SACC for every block; one copy is enough.
        if self._concise_sacc is None:
            self._concise_sacc = super().get_sacc_with_concise_dtypes()
        return self._concise_sacc

    def precompute(self, pairs):
        """
        Build every workspace and covariance workspace needed by ``pairs``.

        Call this before forking workers, which would otherwise each build (and
        discard) their own copies.
        """
        for comb1, comb2 in pairs:
            tracer_names = dict(zip((1, 2, 3, 4), (*comb1, *comb2)))
            if all(t in self.mask_names for t in tracer_names.values()):
                self.get_workspaces_dict(tracer_names, self.binning_info)
                names = self.get_mask_names_dict(tracer_names)
                self.namaster.covariance_workspace_from_masks(*(names[i] for i in range(1, 5)))

    def cleanup(self):
        """Remove the temporary output directory, if this object made one."""
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
