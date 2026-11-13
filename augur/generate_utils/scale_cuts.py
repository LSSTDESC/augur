"""Shared helpers for parsing scale-cut configuration values.

These utilities provide consistent handling of scalar-or-list scale-cut inputs
across probe generators.  A list value is interpreted as one value per
``tracer_combs`` entry, matched by the current combination index.
"""

from numbers import Real


# Conservative pair-combination rule by cut type.
# For lower limits, conservative means larger threshold; for upper limits,
# conservative means smaller threshold.
_PAIR_CONSERVATIVE_REDUCER = {
    'lmax': min,
    'kmax': min,
    'theta_max': min,
    'r_max': min,
    'theta_min': max,
    'r_min': max,
}


def _normalize_combination(comb):
    """Normalize a tracer-combination object to a tuple."""
    try:
        return tuple(comb)
    except TypeError as exc:
        raise ValueError(f"Invalid tracer combination {comb!r}.") from exc


def _coerce_cut_value(value, key_name):
    """Convert a raw scale-cut value into ``float`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == 'none':
            return None
        raise ValueError(
            f"{key_name} must be numeric, None, 'None', or a list of those values."
        )
    if not isinstance(value, Real):
        raise ValueError(
            f"{key_name} must be numeric, None, 'None', or a list of those values."
        )
    return float(value)


def _extract_bin_index(tracer_name):
    """Extract the numeric bin index from a tracer name.

    E.g., 'lens0' → 0, 'src1' → 1, 'lens_ia5' → 5.
    Returns None if the tracer name does not end with a digit.

    Parameters
    ----------
    tracer_name : str
        Tracer name like 'lens0', 'src1', etc.

    Returns
    -------
    int or None
        The trailing bin index, or None if not found.
    """
    # Extract trailing digits from the tracer name
    i = len(tracer_name) - 1
    while i >= 0 and tracer_name[i].isdigit():
        i -= 1
    if i == len(tracer_name) - 1:
        # No trailing digits
        return None
    return int(tracer_name[i + 1:])


def parse_combination_scale_cut(stat_cfg, key_name, comb, tracer_combs_key='tracer_combs'):
    """Resolve a scale-cut value for one tracer combination.

        The config value may be:
            - scalar (applies to all combinations),
            - ``None`` / ``'None'`` (no cut),
            - list with one entry per ``tracer_combs`` element (per-combination mode), or
            - list with one entry per tomographic bin (per-bin mode).

        In per-bin mode, the value for a pair ``(i, j)`` is computed from bin
        entries ``i`` and ``j`` using a conservative reducer:
            - ``max`` for lower limits (``theta_min``, ``r_min``),
            - ``min`` for upper limits (``lmax``, ``kmax``, ``theta_max``, ``r_max``).

        Per-bin mode supports both symmetric square grids and rectangular asymmetric
        grids where lens bins ≠ source bins. Tracer names are parsed to extract
        bin indices (e.g., ``lens0`` → index 0, ``src1`` → index 1).

    Parameters
    ----------
    stat_cfg : dict
        Per-statistic configuration block.
    key_name : str
        Name of the scale-cut key to parse.
    comb : sequence
        Current tracer combination (tuple of tracer names like ('lens0', 'src1')).
    tracer_combs_key : str
        Key holding tracer-combination definitions in ``stat_cfg``.

    Returns
    -------
    float or None
        Parsed scale-cut value for this combination.
    """
    raw_value = stat_cfg.get(key_name, None)

    if isinstance(raw_value, list):
        tracer_combs = stat_cfg.get(tracer_combs_key, [])
        comb_norm = _normalize_combination(comb)

        # Mode 1: explicit per-combination list (legacy behaviour).
        if len(raw_value) == len(tracer_combs):
            tracer_combs_norm = [_normalize_combination(c) for c in tracer_combs]
            if comb_norm not in tracer_combs_norm:
                raise ValueError(
                    f"Tracer combination {comb!r} not found in {tracer_combs_key}."
                )
            raw_value = raw_value[tracer_combs_norm.index(comb_norm)]
        # Mode 2: per-bin list. Combine both bins conservatively.
        # Handle both symmetric (square grid) and asymmetric (rectangular grid) cases.
        elif len(comb_norm) == 2:
            # Try to extract bin indices from tracer names.
            tracer1_name, tracer2_name = comb_norm
            bin_idx1 = _extract_bin_index(tracer1_name)
            bin_idx2 = _extract_bin_index(tracer2_name)

            if bin_idx1 is None or bin_idx2 is None:
                raise ValueError(
                    f"Per-bin list mode for {key_name} could not extract bin indices "
                    f"from tracer names {comb!r}. Names must end with numeric indices."
                )

            # Determine grid structure from all tracer combinations.
            # Extract (tracer_name, bin_idx, family) tuples for all combinations.
            all_tracer_info = []
            for pair in tracer_combs:
                pair_norm = _normalize_combination(pair)
                if len(pair_norm) != 2:
                    continue
                for tracer_name in pair_norm:
                    bin_idx = _extract_bin_index(tracer_name)
                    if bin_idx is not None:
                        all_tracer_info.append((tracer_name,
                                                bin_idx,
                                                tracer_name[:-len(str(bin_idx))]))

            # Group by tracer family and extract unique bins per family.
            family_bins = {}
            family_order = []  # Track order of first appearance
            for tracer_name, bin_idx, family in all_tracer_info:
                if family not in family_bins:
                    family_bins[family] = set()
                    family_order.append(family)
                family_bins[family].add(bin_idx)

            # Compute expected per-bin list length for rectangular grid.
            expected_length = sum(len(family_bins[fam]) for fam in family_order)

            if len(raw_value) == expected_length:
                # Rectangular grid mode: values are organized by family.
                # Build offset mapping: family → starting index in per-bin list.
                family_offset = {}
                offset = 0
                for family in family_order:
                    family_offset[family] = offset
                    offset += len(family_bins[family])

                # Determine which family each tracer belongs to.
                tracer1_family = tracer1_name[:-len(str(bin_idx1))]
                tracer2_family = tracer2_name[:-len(str(bin_idx2))]

                # Look up values in per-bin list.
                if (tracer1_family not in family_offset or tracer2_family not in family_offset):
                    raise ValueError(
                        f"Tracer families {tracer1_family!r}, {tracer2_family!r} from {comb!r} "
                        f"not found in expected families {list(family_offset.keys())!r}."
                    )

                idx1 = family_offset[tracer1_family] + bin_idx1
                idx2 = family_offset[tracer2_family] + bin_idx2

                if idx1 >= len(raw_value) or idx2 >= len(raw_value):
                    raise ValueError(
                        f"Per-bin list for {key_name} has length {len(raw_value)}, "
                        f"but {comb!r} requires indices {idx1} and {idx2}."
                    )

                reducer = _PAIR_CONSERVATIVE_REDUCER.get(key_name, min)
                val_i = _coerce_cut_value(raw_value[idx1], key_name)
                val_j = _coerce_cut_value(raw_value[idx2], key_name)
                if val_i is None and val_j is None:
                    return None
                if val_i is None:
                    return val_j
                if val_j is None:
                    return val_i
                return reducer(val_i, val_j)
            elif len(raw_value) <= 2:
                # Small square grid: might be simple [binA, binB] indexing.
                if bin_idx1 >= len(raw_value) or bin_idx2 >= len(raw_value):
                    raise ValueError(
                        f"Per-bin list for {key_name} has length {len(raw_value)}, "
                        f"but {comb!r} requires indices {bin_idx1} and {bin_idx2}."
                    )
                reducer = _PAIR_CONSERVATIVE_REDUCER.get(key_name, min)
                val_i = _coerce_cut_value(raw_value[bin_idx1], key_name)
                val_j = _coerce_cut_value(raw_value[bin_idx2], key_name)
                if val_i is None and val_j is None:
                    return None
                if val_i is None:
                    return val_j
                if val_j is None:
                    return val_i
                return reducer(val_i, val_j)
            else:
                raise ValueError(
                    f"If {key_name} is a list, it must either have the same length as "
                    f"{tracer_combs_key} (per-combination mode, length {len(tracer_combs)}), "
                    f"or length matching the sum of unique bins per tracer family "
                    f"(per-bin rectangular mode, length {expected_length}). "
                    f"Got length {len(raw_value)}."
                )
        else:
            raise ValueError(
                f"If {key_name} is a list, it must either have the same length as "
                f"{tracer_combs_key} (per-combination mode) or represent per-bin cuts "
                "for 2-bin combinations."
            )

    return _coerce_cut_value(raw_value, key_name)
