"""Helpers that build individual probe sections of the SACC data vector.

Each module here takes the parsed config plus the partially-built SACC object and
returns the firecrown statistics and two-point filters for one probe, so that
``augur.generate`` stays a thin orchestrator.
"""

from augur.generate_utils.cmb_lensing import add_cmb_lensing, get_cmb_noise

__all__ = ['add_cmb_lensing', 'get_cmb_noise']
