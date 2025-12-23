from pathlib import Path
from augur.analyze import Analyze


def test_analyze():
    base_path = Path(__file__).parent
    fish = Analyze(f'{base_path}/test.yaml')
    # Test derivatives
    fish.get_derivatives(method='5pt_stencil')
    fish.get_fisher_matrix()
    fish.get_fisher_bias()
    # Test alternative derivative methods
    fish.get_derivatives(method='numdifftools', force=True)
    try:
        import derivkit
        derivkit.__version__
        fish.get_derivatives(method='derivkit', force=True)
    except ImportError:
        print("derivkit not installed, skipping...")
