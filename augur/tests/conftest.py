import pytest
from ..parser import parse


@pytest.fixture
def example_yaml(tmpdir):
    yaml = parse("examples/srd_v1_3x2.yaml")
    yaml['verbose'] = True
    yaml['generate']['two_point']['sacc_file'] = str(tmpdir.join('test.sacc'))
    yaml['analyze']['two_point']['sacc_data'] = str(tmpdir.join('test.sacc'))
    yaml['analyze']['cosmosis']['output_dir'] = str(tmpdir)

    # make fewer parameters to enable fisher to run faster
    pars = yaml['analyze']['cosmosis']['parameters']
    for key in list(pars.keys()):  # to avoid dictionary size change
        if key not in ['sigma8', 'bias_lens0']:
            del pars[key]

    return yaml
