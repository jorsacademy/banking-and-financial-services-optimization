import numpy as np

from banking_optimization.liquidity_funding import default_problem, solve


def test_funding_identity_holds():
    p = default_problem()
    r = solve(p)

    assert np.isclose(
        r.funding_mix.sum(),
        p.core_funding_need + r.liquidity_buffer,
    )


def test_buffer_covers_all_stress_horizons():
    r = solve(default_problem())
    assert (r.coverage_surplus >= -1e-7).all()


def test_stable_funding_requirement_holds():
    p = default_problem()
    r = solve(p)
    assert r.stable_funding_share >= p.minimum_stable_funding_share - 1e-8


def test_funding_source_bounds_hold():
    p = default_problem()
    r = solve(p)

    for source, value in r.funding_mix.items():
        assert value >= p.funding_sources.loc[source, "minimum"] - 1e-7
        assert value <= p.funding_sources.loc[source, "maximum"] + 1e-7
