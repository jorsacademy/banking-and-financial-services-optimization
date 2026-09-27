from banking_optimization.alm_experiments import sensitivity_grid


def test_sensitivity_grid_has_expected_shape_and_metrics():
    result = sensitivity_grid(
        duration_gaps=(0.70, 0.90),
        shortfall_penalties=(1.0, 3.0),
    )

    assert len(result) == 4
    assert {
        "adaptive_expected_nii",
        "static_expected_nii",
        "adaptive_expected_shortfall",
        "static_expected_shortfall",
        "value_of_adaptivity",
    }.issubset(result.columns)
    assert (result["value_of_adaptivity"] >= -1e-7).all()
