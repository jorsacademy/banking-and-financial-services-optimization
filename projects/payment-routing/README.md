# Payment Routing Optimization

A mixed-integer assignment model for routing synthetic payment segments across processors.

Each route has different fees, success probability, latency, capacity, and eligibility. The cheapest nominal processor is therefore not necessarily the cheapest expected route.

## Objective

Minimize expected processing cost:

- processor fee;
- expected failure cost;
- latency penalty.

## Constraints

- every payment segment is assigned to exactly one eligible route;
- processor capacities cannot be exceeded.

## Run

```bash
python -m banking_optimization.payment_routing
pytest tests/test_payment_routing.py
```

## Extension path

Transaction-level online routing, contextual bandits, smart retries, acquirer cascading, interchange economics, fraud controls, currency constraints, merchant preferences, and latency/service-level distributions.

## Limitations

Synthetic educational model only. It is not a production payment-routing or authorization system.
