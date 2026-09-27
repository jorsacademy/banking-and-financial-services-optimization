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


# 22. Full multistage scenario-tree IRP

The multistage formulation replaces scenario-indexed future decisions with
**node-indexed decisions**.

Let:

- \(N\): all scenario-tree nodes;
- \(N^D\): decision nodes;
- \(N^R\): realized-demand/state nodes;
- \(L\): terminal leaves;
- \(\pi(n)\): parent of node \(n\);
- \(p_n\): probability of reaching node \(n\).

A node represents one information state. Therefore all scenario paths with the
same observed history share the same node and the same decision variables.

This is structural non-anticipativity.

## 22.1 Timing convention

The root is the state before day-1 demand.

At a decision node \(n \in N^D\), routes and delivery quantities are selected.

Demand then realizes and the process moves to one child state node.

At the child node, inventory and cash-out are observed before the next routing
decision is made.

For a three-day binary tree, decisions occur at:

\`\`\`text
stage 0: 1 root decision
stage 1: 2 decisions
stage 2: 4 decisions
\`\`\`

while stage 3 contains eight terminal realized states.

## 22.2 Node-indexed routing decisions

For decision node \(n\) and route \(r\):

\[
z_{nr}\in\{0,1\}
\]

selects route \(r\).

For cashpoint \(i\in I_r\):

\[
q_{nri}\ge0
\]

is the amount delivered on that route.

There is only one \(z_{nr}\) and one \(q_{nri}\) for a given information state,
regardless of how many terminal paths descend from that node.

## 22.3 State transition on tree edges

For realized node \(m\) with parent decision node \(n=\pi(m)\):

\[
B_{mi}
=
B_{ni}
+
\sum_{r:i\in I_r}q_{nri}
+
l_{mi}
-
\tilde d_{mi}
\]

where \(B_{ni}\) is replaced by initial cash when \(n\) is the root.

The cash-out variable is:

\[
l_{mi}\ge0
\]

and receives a high penalty.

## 22.4 Expected operating cost

Let \(C_n^{route}\) and \(C_n^{delivery}\) denote costs incurred at decision node
\(n\).

Let state cost at realized node \(m\) contain holding, safety-shortfall, and
cash-out terms.

The risk-neutral expected objective is:

\[
\mathbb{E}[C]
=
\sum_{n\in N^D}
p_n
\left(
C_n^{route}
+
C_n^{delivery}
\right)
+
\sum_{m\in N^R}
p_m
C_m^{state}
\]

This is equivalent to the probability-weighted cost of terminal paths when the
tree probabilities are consistent.

# 23. Terminal path cost

For terminal leaf \(\ell\), define the unique root-to-leaf path:

\[
P(\ell)
\]

The total cost on that path is:

\[
C_\ell
=
\sum_{n\in P(\ell)\cap N^D}
\left(
C_n^{route}
+
C_n^{delivery}
\right)
+
\sum_{m\in P(\ell)\cap N^R}
C_m^{state}
\]

These leaf costs define the tail-risk distribution.

# 24. CVaR risk aversion

Let:

- \(\alpha\in(0,1)\): CVaR confidence level;
- \(\eta\): VaR threshold variable;
- \(\xi_\ell\ge0\): excess loss for terminal leaf \(\ell\).

For every terminal leaf:

\[
\xi_\ell
\ge
C_\ell-\eta
\]

The standard Rockafellar-Uryasev CVaR representation is:

\[
CVaR_\alpha(C)
=
\eta
+
\frac{1}{1-\alpha}
\sum_{\ell\in L}
p_\ell\xi_\ell
\]

The risk-averse objective is:

\[
\min
\quad
\mathbb{E}[C]
+
\lambda
CVaR_\alpha(C)
\]

where:

\[
\lambda\ge0
\]

controls risk aversion.

When \(\lambda=0\), the formulation is risk-neutral.

As \(\lambda\) increases, the optimizer can accept higher expected operating
cost in exchange for lower tail cost.

# 25. Risk-neutral vs risk-averse comparison

The project solves both policies on the exact same scenario tree.

Reported metrics include:

- expected cost;
- CVaR cost;
- worst terminal-path cost;
- expected cash-out;
- root-stage replenishment;
- probability-weighted route usage.

The key trade-offs are:

\[
\Delta E
=
E[C]_{risk\ averse}
-
E[C]_{risk\ neutral}
\]

and:

\[
\Delta CVaR
=
CVaR_{risk\ neutral}
-
CVaR_{risk\ averse}
\]

A positive \(\Delta CVaR\) means the risk-averse policy reduced tail cost.

# 26. Relationship to rolling-horizon stochastic IRP

The two stochastic layers answer different questions.

The rolling-horizon model uses:

\`\`\`text
two-stage scenario recourse
+
receding-horizon re-optimization
\`\`\`

The multistage model uses:

\`\`\`text
explicit scenario tree
+
history-dependent node decisions
+
full tree non-anticipativity
+
CVaR terminal-path risk
\`\`\`

The multistage formulation is stronger conceptually but grows exponentially
with branching depth.

For larger instances, likely solution methods include:

- scenario reduction;
- progressive hedging;
- nested Benders decomposition;
- stochastic dual dynamic programming for compatible relaxations;
- branch-and-price with scenario decomposition;
- approximate dynamic programming / policy approximation.


# 27. Scenario reduction

Let terminal scenarios be indexed by \(\omega\in\Omega\), with probabilities
\(p_\omega\).

Each terminal scenario is represented by a flattened demand-path vector:

\[
v_\omega
=
(
d_{\omega,1,1},
\ldots,
d_{\omega,T,|I|}
)
\]

covering all stages and cashpoints.

Each dimension is standardized using probability-weighted mean and variance.
Distance between terminal scenarios is then:

\[
D(\omega,\omega')
=
\left\|
\tilde v_\omega
-
\tilde v_{\omega'}
\right\|_2
\]

A forward-selection set \(S\subseteq\Omega\) is built greedily.

At each selection step, candidate scenario \(j\) is evaluated by:

\[
\Phi(S\cup\{j\})
=
\sum_{\omega\in\Omega}
p_\omega
\min_{s\in S\cup\{j\}}
D(\omega,s)
\]

The candidate producing the smallest probability-weighted distortion is added.

After selecting the target number of representative leaves, every original
scenario is assigned to its nearest retained scenario:

\[
a(\omega)
=
\arg\min_{s\in S}
D(\omega,s)
\]

The reduced probability of representative \(s\) is:

\[
\hat p_s
=
\sum_{\omega:a(\omega)=s}
p_\omega
\]

so total probability is preserved:

\[
\sum_{s\in S}\hat p_s=1
\]

The reduced multistage tree is reconstructed from the retained root-to-leaf
histories. Internal-node probability equals the total probability of retained
leaves descending from that node.

The reported reduction distortion is:

\[
\mathcal{D}
=
\sum_{\omega\in\Omega}
p_\omega
D(\omega,a(\omega))
\]

This is a scenario-representation diagnostic, not an optimization-error bound.

# 28. Progressive Hedging decomposition

The extensive-form multistage model couples terminal scenarios through
non-anticipative decisions at shared history nodes.

Progressive Hedging decomposes the model by terminal scenario while
coordinating those shared decisions iteratively.

Let:

- \(\omega\): terminal scenario;
- \(n\): information-history node appearing on scenario \(\omega\)'s path;
- \(x_{\omega n}\): scenario-copy decision vector at node \(n\);
- \(p_\omega\): scenario probability;
- \(\bar x_n\): consensus decision at node \(n\).

The probability-weighted consensus is:

\[
\bar x_n
=
\frac{
\sum_{\omega:n\in P(\omega)}
p_\omega x_{\omega n}
}{
\sum_{\omega:n\in P(\omega)}
p_\omega
}
\]

Only scenarios sharing history node \(n\) participate in that consensus.

## 28.1 Scenario subproblem

Each scenario solves its own path MILP:

\[
\min
\quad
C_\omega(x_\omega)
+
w_\omega^\top x_\omega
+
\rho
\|x_\omega-\bar x\|_1
\]

subject to the physical inventory-routing constraints on that scenario path.

The base cost \(C_\omega\) includes:

- route cost;
- distance cost;
- handling cost;
- inventory holding;
- safety shortfall;
- cash-out penalty.

## 28.2 Why L1 instead of quadratic PH

Classical Progressive Hedging uses:

\[
\frac{\rho}{2}
\|x_\omega-\bar x\|_2^2
\]

The project uses SciPy/HiGHS linear MILPs, so the proximal term is replaced by
an L1 penalty.

For binary route variable \(z\) and fixed consensus \(\bar z\):

\[
|z-\bar z|
=
z(1-2\bar z)
+
\bar z
\]

Since \(\bar z\) is constant during a scenario solve, the variable part is
linear.

For continuous delivery \(q\), introduce:

\[
d^+_{\omega n i}\ge0,
\qquad
d^-_{\omega n i}\ge0
\]

with:

\[
q_{\omega n i}
-
d^+_{\omega n i}
+
d^-_{\omega n i}
=
\bar q_{ni}
\]

Then:

\[
|q_{\omega n i}-\bar q_{ni}|
=
d^+_{\omega n i}
+
d^-_{\omega n i}
\]

at optimum.

This is therefore an **L1 PH variant / decomposition heuristic**, not the
standard quadratic PH algorithm.

## 28.3 Multiplier update

After solving all leaf subproblems and recomputing consensus, multipliers are
updated as:

\[
w_{\omega n}^{k+1}
=
w_{\omega n}^{k}
+
\rho
\left(
x_{\omega n}^{k+1}
-
\bar x_n^{k+1}
\right)
\]

Separate penalty magnitudes are used for route binaries and delivery quantities
because their numerical scales differ.

## 28.4 Consensus residual

Route residual is the maximum absolute disagreement:

\[
r_z
=
\max_{\omega,n,r}
|z_{\omega nr}-\bar z_{nr}|
\]

Delivery disagreement is normalized by cashpoint maximum delivery:

\[
r_q
=
\max_{\omega,n,i}
\frac{
|q_{\omega ni}-\bar q_{ni}|
}{
Q_i
}
\]

The overall reported residual is:

\[
r
=
\max(r_z,r_q)
\]

The PH process is marked converged when:

\[
r\le\epsilon
\]

for the configured tolerance.

Because integer route decisions can oscillate, convergence is not guaranteed
for every parameterization.

# 29. Reduced-tree exact benchmark

The scaling workflow is:

\`\`\`text
full scenario tree
        ↓
scenario reduction
        ↓
reduced scenario tree
        ├── exact extensive-form MILP
        └── L1 Progressive Hedging
\`\`\`

The exact reduced-tree model provides a validation reference.

Before non-anticipativity residual reaches zero, the PH scenario-wise expected
cost can be lower than the extensive-form optimum because scenario copies still
have residual disagreement.

Therefore:

\[
C_{PH}-C_{exact}
\]

is reported only as a cost difference diagnostic and is not automatically
called an optimality gap.

# 30. Computational interpretation

Scenario reduction attacks the number of uncertainty paths.

Progressive Hedging attacks the coupling across scenarios.

They address different sources of complexity and can be combined:

\[
\text{large tree}
\rightarrow
\text{representative reduced tree}
\rightarrow
\text{parallelizable scenario subproblems}
\]

For larger implementations, natural extensions include:

- parallel leaf-subproblem solves;
- asynchronous PH;
- adaptive penalty updates;
- fixing stable binary decisions;
- branch-and-price inside route subproblems;
- nested decomposition for CVaR;
- empirical scenario clustering;
- scenario reduction with Wasserstein or transportation metrics.
