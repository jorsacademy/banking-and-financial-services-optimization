# IRRBB Curve-Hedge Optimization

A synthetic banking-book interest-rate-risk project that chooses a bounded
portfolio of swaps against a key-rate PV01 exposure.

The project is deliberately decision-oriented. It does not stop at measuring a
duration gap or reporting an EVE sensitivity table. It solves for the hedge
portfolio that reduces the worst EVE loss across a set of stylized yield-curve
shocks while charging the optimizer for hedge usage.

## State representation

The banking book is summarized by a key-rate PV01 vector across 2Y, 5Y, 10Y
and 20Y tenors.

Each candidate payer swap contributes its own key-rate PV01 vector. A signed
hedge position therefore changes the residual curve exposure:

~~~text
residual PV01
=
banking-book PV01
+
hedge sensitivity matrix × hedge notionals
~~~

## Curve scenarios

The synthetic stress set contains:

- parallel up;
- parallel down;
- short-end shock;
- long-end shock;
- steepener;
- flattener.

For scenario s, first-order EVE change is approximated by:

~~~text
Delta EVE_s = - residual_PV01 dot shock_s
~~~

The model is intentionally a transparent first-order approximation rather than
a full cash-flow revaluation engine.

## Optimization model

The decision variables are signed hedge notionals.

The LP minimizes:

~~~text
worst scenario EVE loss
+
hedge-cost weight × gross hedge cost
~~~

subject to:

- instrument-level hedge limits;
- a total gross hedge-notional budget;
- one common hedge portfolio across all stress scenarios.

Positive and negative hedge notionals are represented with standard positive /
negative LP variables, so absolute hedge usage remains linear.

## Why this matters

The existing ALM project chooses assets, funding and a coarse duration hedge.
This extension moves to a more treasury-style curve-risk representation:

~~~text
balance-sheet policy
        ↓
key-rate PV01
        ↓
non-parallel curve shocks
        ↓
optimal swap hedge
        ↓
residual EVE tail loss
~~~

That makes the hedge decision sensitive to curve shape rather than only to one
scalar duration gap.

## Run

~~~bash
python -m banking_optimization.irrbb
python projects/irrbb-curve-hedging/run.py
~~~

The project runner exports:

~~~text
outputs/
├── optimal_hedge_notionals.csv
├── residual_key_rate_pv01.csv
├── scenario_eve_change.csv
└── hedge_comparison.csv
~~~

## Validation

The tests verify:

- the optimized hedge reduces worst-case synthetic EVE loss;
- gross and instrument hedge limits are respected;
- residual key-rate PV01 reconciles exactly to the base exposure plus hedge
  sensitivities;
- scenario EVE values reconcile to the residual PV01 and shock vectors.

## Limitations

This is an educational research model, not a regulatory IRRBB, EVE, NII,
hedge-accounting or treasury system.

A production implementation would require instrument-level contractual cash
flows, curve construction, repricing conventions, optionality, behavioral
deposit and prepayment models, basis risk, currencies, valuation adjustments,
accounting treatment, institution-specific limits and independent model
validation.
