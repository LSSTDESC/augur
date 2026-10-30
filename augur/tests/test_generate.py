from pathlib import Path

import sacc

from augur.generate import generate
from augur.utils.config_io import parse_config


def test_generate():
    base_path = Path(__file__).parent
    generate(f'{base_path}/test.yaml')


def test_generate_use_sacc_round_trip(tmp_path):
    """Feed a generated sacc back in through `use_sacc`.

    This path had no coverage at all, which is how three separate bugs survived
    in it: a data-type scoping error in the filter builder, the sacc tracer
    object being passed where firecrown wants its name, and the cosmology
    parameters being read from the config block rather than the built cosmology.
    """
    base_path = Path(__file__).parent
    config = parse_config(f'{base_path}/test.yaml')
    sacc_path = tmp_path / 'round_trip.sacc'
    config['fiducial_sacc_path'] = str(sacc_path)

    generate(config)
    S = sacc.Sacc.load_fits(str(sacc_path))

    lk, _, _ = generate(parse_config(f'{base_path}/test.yaml'), return_all_outputs=True,
                        write_sacc=False, use_sacc=S, sacc_path=str(sacc_path))

    # The likelihood must cover every data type present in the file.
    types = {st.statistic.sacc_data_type for st in lk.statistics}
    assert types == set(S.get_data_types())
