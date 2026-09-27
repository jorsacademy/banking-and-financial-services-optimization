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
