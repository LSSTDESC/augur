import pytest
from ..parser import parse

@pytest.fixture
def example_yaml():
    yaml =  parse("examples/srd_v1_3x2.yaml")
    yaml ['verbose'] = True
    return yaml
