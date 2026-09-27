# Mathematical Model

The project contains two optimization architectures.

The staged model separates replenishment planning from route construction.
The joint model is a small exact Inventory Routing Problem (IRP) in which
route selection and delivered cash are optimized simultaneously.

# 1. Core sets

Let:

- (I): cashpoints;
- (T): planning days;
- (G): operational/geographic clusters;
- (R): candidate CIT routes.

For route (r):

- (I_r subseteq I): cashpoints visited by route (r);
- (D_r): route distance;
- (V_r): sum of cashpoint visit costs on the route.

Only routes with at most the configured stop limit are included.

# 2. Parameters

For cashpoint (i) and day (t):

- (d_{it}): forecast net cash withdrawal;
- (B_i^0): initial cash;
- (C_i): cash capacity;
- (S_i): safety-stock target;
- (Q_i): maximum delivery on one visit.

System parameters:

- (U_t): vault dispatch limit;
- (K): CIT vehicle cash capacity;
- (ar V): maximum routes/vehicles per day;
- (L_g): daily visit limit for cluster (g);
- (c^H): cash handling cost per unit;
- (c^I): idle-cash holding cost per unit-day;
- (c^F): fixed cost per operated route/vehicle;
- (c^D): distance cost per route-distance unit;
- (c^S): safety-stock shortfall penalty.

# 3. Route generation

For every cashpoint subset with size at most the stop limit, the minimum-distance
depot tour is solved exactly by enumerating stop permutations.

This creates a route catalog.

For six cashpoints and at most three stops:

[
|R|
=
{6 choose 1}
+
{6 choose 2}
+
{6 choose 3}
=
41
]

candidate routes.

The integrated optimization selects from these route columns.

# 4. Joint IRP decision variables

For every day (t) and route (r):

[
z_{tr}in{0,1}
]

equals one if route (r) is operated.

For each (iin I_r):

[
q_{tri}ge0
]

is cash delivered to cashpoint (i) by route (r) on day (t).

For each cashpoint/day:

- (B_{it}ge0): end-of-day cash balance;
- (s_{it}ge0): safety-stock shortfall.

# 5. Joint objective

The integrated model minimizes:

[
sum_{t,r}
left(
c^F + V_r + c^D D_r
ight)z_{tr}
]

[
+
sum_{t,r}sum_{iin I_r}
c^H q_{tri}
+
sum_{i,t}
left(
c^I B_{it}
+
c^S s_{it}
ight)
]

Route distance therefore affects the replenishment policy directly.

# 6. Inventory conservation

For day 1:

[
B_{i1}
=
B_i^0
+
sum_{r:iin I_r}q_{1ri}
-
d_{i1}
]

For later days:

[
B_{it}
=
B_{i,t-1}
+
sum_{r:iin I_r}q_{tri}
-
d_{it}
]

# 7. Cashpoint route assignment

Each cashpoint may appear in at most one selected route per day:

[
sum_{r:iin I_r}z_{tr}
le1
]

# 8. Delivery-route linking

For every candidate route containing cashpoint (i):

[
q_{tri}
le
Q_i z_{tr}
]

A route-specific delivery can be positive only when the route is operated.

# 9. Route vehicle capacity

For every route/day:

[
sum_{iin I_r}q_{tri}
le
Kz_{tr}
]

This is stronger than an aggregate daily fleet-capacity approximation because
cash capacity is enforced route by route.

# 10. Safety-stock target

[
B_{it}+s_{it}ge S_i
]

The target is soft: violations remain feasible but are penalized.

# 11. Cashpoint capacity

Before forecast withdrawals:

[
B_{i,t-1}
+
sum_{r:iin I_r}q_{tri}
le
C_i
]

with initial cash replacing (B_{i,t-1}) on the first day.

# 12. Vault dispatch capacity

[
sum_rsum_{iin I_r}q_{tri}
le
U_t
]

# 13. Daily route/fleet availability

[
sum_r z_{tr}
le
ar V
]

Every selected route represents one deployed CIT vehicle trip.

# 14. Cluster visit limits

For cluster (g):

[
sum_r
|I_rcap g|
z_{tr}
le
L_g
]

Because a cashpoint may belong to at most one selected route on a day, the
coefficient counts actual visits.

# 15. Staged baseline

The staged baseline first solves a replenishment/fleet MILP without route
distance in the objective.

A second exact routing procedure then partitions the selected daily visits into
capacity-feasible routes and minimizes distance.

For comparison, the staged integrated cost is calculated as:

[
C_{staged}^{full}
=
C_{staged}^{planning}
+
c^D D_{staged}
]

where (D_{staged}) is the exact post-optimization route distance.

# 16. Joint-vs-staged comparison

The reported improvement is:

[
Delta C
=
C_{staged}^{full}
-
C_{joint}
]

A nonnegative value indicates that accounting for route economics inside the
replenishment optimization improved or matched the staged policy under the same
synthetic assumptions.

# 17. Monte Carlo validation

The deterministic forecast is perturbed multiplicatively:

[
	ilde d_{it}
=
d_{it}epsilon_{it}
]

with:

[
epsilon_{it}
sim
operatorname{Lognormal}
left(
-rac{sigma^2}{2},
sigma
ight)
]

The optimized deliveries are held fixed during a simulation replication.

Reported robustness metrics include:

- mean cash-out volume;
- 95th-percentile cash-out volume;
- mean idle cash;
- mean service-event rate.

# 18. Modeling scope

The route-column model is exact for the small synthetic instance but does not
scale by brute-force route enumeration to large networks.

A larger implementation would typically require column generation,
branch-and-price, decomposition, rolling-horizon methods, or route-generation
heuristics.


# 19. Stochastic rolling-horizon IRP

The rolling-horizon layer embeds a two-stage stochastic route-column model
inside a receding-horizon controller.

At a replanning date, let:

- (H): short look-ahead horizon;
- (Omega): demand scenarios;
- (p_omega): scenario probability;
- (t=1): current execution day;
- (t=2,ldots,H): future recourse days.

## 19.1 First-stage decisions

Today's decisions are shared across all scenarios:

- (z_r^0 in {0,1}): whether route (r) operates today;
- (q_{ri}^0 ge 0): cash delivered today to cashpoint (i) on route (r).

These variables are **non-anticipative** because they are not indexed by
scenario.

They must be chosen before realized demand is known.

## 19.2 Scenario-specific recourse

For future days and scenario (omega):

- (z_{omega tr} in {0,1}): future route selection;
- (q_{omega tri} ge 0): future route-specific delivery;
- (B_{omega it} ge 0): end-of-day inventory;
- (s_{omega it} ge 0): safety-stock shortfall;
- (l_{omega it} ge 0): lost demand / cash-out.

Future routing and replenishment decisions can react to the scenario.

This produces a two-stage approximation rather than a full multistage
scenario-tree model.

## 19.3 Inventory balance with lost demand

For the current day:

[
B_{omega i1}
=
B_i^{current}
+
sum_{r:iin I_r}q_{ri}^0
+
l_{omega i1}
-
	ilde d_{omega i1}
]

For later horizon days:

[
B_{omega it}
=
B_{omega i,t-1}
+
sum_{r:iin I_r}q_{omega tri}
+
l_{omega it}
-
	ilde d_{omega it}
]

Lost demand prevents extreme demand scenarios from making the model infeasible.

It is bounded by realized scenario demand and receives a large penalty.

## 19.4 Expected-cost objective

The first-stage route and handling costs are paid once.

Future costs and state costs are probability weighted:

[
min
C^{first}(z^0,q^0)
+
sum_{omegainOmega}
p_omega
left[
C_omega^{recourse}
+
C_omega^{inventory}
+
C_omega^{safety}
+
C_omega^{cashout}
ight]
]

where cash-out cost is:

[
C_omega^{cashout}
=
c^L
sum_{i,t}l_{omega it}
]

and (c^L) is the explicit lost-demand penalty.

## 19.5 First-stage operational constraints

Today's shared route variables satisfy the same physical restrictions as the
deterministic joint IRP:

- route-specific vehicle cash capacity;
- cashpoint maximum delivery;
- at most one route visit per cashpoint;
- vault dispatch capacity;
- maximum daily vehicles/routes;
- cluster visit limits;
- cashpoint storage capacity.

Since these decisions are shared across all scenarios, the same physical plan
must be executable regardless of which demand realization occurs.

## 19.6 Scenario-specific future constraints

For each scenario and future horizon day, the model enforces:

- route-specific vehicle capacity;
- route/delivery linking;
- at most one route per cashpoint/day;
- vault capacity;
- fleet availability;
- cluster visit limits;
- cashpoint capacity;
- safety-stock accounting;
- lost-demand bounds.

## 19.7 Receding-horizon execution

The scenario-specific future decisions are not executed directly.

At day (k):

1. observe current inventory;
2. generate scenarios for days (k,ldots,k+H-1);
3. solve the stochastic IRP;
4. execute only (z^0,q^0);
5. observe actual demand;
6. update inventory and realized cash-out;
7. shift to day (k+1);
8. generate new scenarios and solve again.

Therefore the operational policy is:

[
pi(S_k)
ightarrow
a_k
ightarrow
W_k
ightarrow
S_{k+1}
]

where:

- (S_k): observed cash state;
- (a_k): executed route/replenishment decision;
- (W_k): realized demand uncertainty;
- (S_{k+1}): updated state.

## 19.8 Deterministic rolling benchmark

A deterministic rolling policy uses the same state-update and receding-horizon
architecture but solves the deterministic joint IRP using point forecasts.

Both policies are evaluated on the same out-of-sample realized demand path.

The comparison reports:

[
Delta C =
C_{deterministic rolling}
-
C_{stochastic rolling}
]

and:

[
Delta L =
L_{deterministic rolling}
-
L_{stochastic rolling}
]

for realized cost and realized cash-out respectively.

Unlike the deterministic staged-vs-joint benchmark, these quantities are
path-dependent. A stochastic policy is not expected to dominate on every
single realization; its purpose is to improve decision quality under repeated
uncertainty across a distribution of possible demand paths.

# 20. Scenario generation

Scenario demand uses multiplicative lognormal forecast error:

[
	ilde d_{omega it}
=
d_{it}epsilon_{omega it}
]

with:

[
epsilon_{omega it}
sim
Lognormal
left(
-rac{sigma^2}{2},
sigma
ight)
]

The implementation currently uses independent synthetic multipliers.

Research extensions include correlated cashpoint shocks, common demand
factors, regime-switching volatility, scenario reduction, empirical residual
bootstrap, and distributionally robust ambiguity sets.

# 21. Current stochastic scope

The model is deliberately a **two-stage stochastic approximation inside a
rolling horizon**.

It does not yet impose multistage non-anticipativity among future scenario
branches.

A full multistage version would require scenario-tree nodes or policy-based
recourse so that scenarios sharing the same history also share decisions until
they diverge.
