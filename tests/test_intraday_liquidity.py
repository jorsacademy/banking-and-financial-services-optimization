from banking_optimization.intraday_liquidity import default_problem, solve


def test_every_payment_is_scheduled_once_and_within_window():
    p = default_problem()
    r = solve(p)

    counts = r.schedule["payment_id"].value_counts()
    assert set(counts.index) == set(p.payments.index)
    assert (counts == 1).all()

    for row in r.schedule.itertuples(index=False):
        earliest = p.payments.loc[row.payment_id, "earliest_slot"]
        deadline = p.payments.loc[row.payment_id, "deadline_slot"]
        assert earliest <= row.slot <= deadline


def test_intraday_liquidity_never_negative():
    r = solve(default_problem())
    assert (r.liquidity_profile >= -1e-7).all()


def test_slot_capacity_holds():
    p = default_problem()
    r = solve(p)
    counts = r.schedule.groupby("slot").size()
    assert (counts <= p.max_payments_per_slot).all()
