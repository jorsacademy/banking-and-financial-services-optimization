# Mathematical Model

## Sets

Let:

- (I): cashpoints;
- (T): planning days;
- (G): geographic/operational clusters.

## Parameters

For cashpoint (i) and day (t):

- (d_{it}): forecast net cash withdrawal;
- (I_i^0): initial cash;
- (C_i): cash capacity;
- (S_i): safety-stock target;
- (Q_i): maximum replenishment per visit;
- (c_i^V): cashpoint visit cost.

System parameters:

- (D_t): central-vault dispatch capacity;
- (K): cash capacity per CIT vehicle;
- (M): maximum stops per vehicle;
- (ar V): maximum available vehicles per day;
- (L_g): maximum daily visits in cluster (g);
- (c^H): cash handling cost per unit;
- (c^I): idle-cash holding cost per unit-day;
- (c^F): fixed cost per deployed vehicle;
- (c^S): service-shortfall penalty.

## Decision variables

For each cashpoint and day:

- (q_{it} ge 0): replenishment quantity;
- (y_{it} in {0,1}): whether the cashpoint is visited;
- (B_{it} ge 0): end-of-day cash balance;
- (s_{it} ge 0): safety-stock shortfall.

For each day:

- (v_t in mathbb{Z}_+): deployed CIT vehicles.

## Objective

[
min
sum_{i,t}
left(
c^H q_{it}
+
c_i^V y_{it}
+
c^I B_{it}
+
c^S s_{it}
ight)
+
sum_t c^F v_t
]

The objective captures the trade-off between frequent replenishment, idle cash, fleet use, and service risk.

## Inventory conservation

For the first day:

[
B_{i1}
=
I_i^0 + q_{i1} - d_{i1}
]

For later days:

[
B_{it}
=
B_{i,t-1} + q_{it} - d_{it}
]

## Visit linking

[
q_{it} le Q_i y_{it}
]

A replenishment is possible only when a visit is opened.

## Safety-stock target

[
B_{it} + s_{it} ge S_i
]

Safety stock is soft rather than hard. Shortfalls remain feasible but receive a large penalty.

## Cashpoint capacity

Before daily withdrawals:

[
B_{i,t-1} + q_{it} le C_i
]

with initial cash replacing (B_{i,t-1}) on day 1.

End-of-day balance is also bounded by capacity.

## Central-vault dispatch capacity

[
sum_i q_{it} le D_t
]

## Fleet carrying capacity

[
sum_i q_{it} le K v_t
]

## Vehicle stop capacity

[
sum_i y_{it} le M v_t
]

## Fleet availability

[
0 le v_t le ar V
]

with integer (v_t).

## Cluster visit limits

For every cluster (g):

[
sum_{i in g} y_{it} le L_g
]

These constraints approximate local operational or routing capacity before detailed routes are constructed.

# Routing layer

Once daily visits are selected, routing is solved over the active cashpoints.

For each candidate route (r):

- total delivery load must not exceed (K);
- number of stops must not exceed (M);
- route starts and ends at the depot.

For small instances, all feasible stop subsets are enumerated. The best stop order for each subset is found exactly, then dynamic programming selects the minimum-distance partition of all active cashpoints into at most (v_t) routes.

This routing layer is a post-optimization operational decomposition, not a fully integrated inventory-routing formulation.

# Simulation layer

Let deterministic forecast demand be (d_{it}).

Realized demand in simulation is:

[
	ilde d_{it}
=
d_{it}epsilon_{it}
]

where:

[
epsilon_{it}
sim
	ext{Lognormal}
left(
-rac{sigma^2}{2},
sigma
ight)
]

so that the multiplicative factor is centered near one.

The optimized replenishment quantities are held fixed during each replication. Realized cash-out volume and idle cash are then measured across replications.

This provides an out-of-sample robustness diagnostic for the deterministic plan.
