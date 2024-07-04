from pathlib import Path
from augur.analyze import Analyze


def test_analyze():
    base_path = Path(__file__).parent
    fish = Analyze(f'{base_path}/test.yaml')
    fish.get_derivatives()
    fish.get_fisher_matrix()
    fish.get_fisher_bias()
