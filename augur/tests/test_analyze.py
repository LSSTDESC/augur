from augur.analyze import Analyze

def test_analyze():
    fish = Analyze('./examples/config_test.yml')
    ders = fish.get_derivatives()
    fishmat = fish.get_fisher_matrix()
    b_vec = fish.compute_fisher_bias()
    