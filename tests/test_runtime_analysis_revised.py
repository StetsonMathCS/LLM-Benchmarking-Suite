from benchmarks.dimensions.runtime_analysis import combined_efficiency_score, relative_efficiency_component


def test_relative_efficiency_equal_or_better_is_one():
    assert relative_efficiency_component(10, 10) == 1
    assert relative_efficiency_component(10, 5) == 1


def test_relative_efficiency_regression_is_ratio():
    assert relative_efficiency_component(10, 20) == 0.5


def test_geometric_mean_and_unavailable_handling():
    assert combined_efficiency_score(1.0, 0.25) == 0.5
    assert combined_efficiency_score(1.0, None) is None
    assert relative_efficiency_component(0, 1) is None

