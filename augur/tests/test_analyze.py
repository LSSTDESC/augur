from augur.analyze import Analyze


def test_analyze():
    fish = Analyze('./examples/config_test.yml')
    fish.get_derivatives()
    fish.get_fisher_matrix()
    fish.get_fisher_bias()
