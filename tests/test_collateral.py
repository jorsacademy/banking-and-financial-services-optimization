from banking_optimization.collateral import default_problem, solve


def test_collateral_covers_each_obligation():
    p = default_problem()
    r = solve(p)

    for obligation, required in p.obligations.items():
        assert r.effective_coverage.loc[obligation] >= required - 1e-7


def test_collateral_respects_inventory():
    p = default_problem()
    r = solve(p)

    used = r.allocation.sum(axis=1)
    for asset, inventory in p.collateral["inventory"].items():
        assert used.loc[asset] <= inventory + 1e-7


def test_collateral_respects_eligibility():
    p = default_problem()
    r = solve(p)

    for asset in p.collateral.index:
        for obligation in p.obligations.index:
            if p.eligibility.loc[asset, obligation] == 0:
                assert abs(r.allocation.loc[asset, obligation]) <= 1e-8
