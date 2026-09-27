# Mathematical Model

This document describes both ALM formulations in the project.

## 1. One-period ALM

Let:

- (A): asset buckets;
- (F): funding sources;
- (S): scenarios;
- (x_a): allocation to asset (a);
- (y_f): amount raised from funding source (f);
- (h^+, h^-): positive and negative hedge notionals;
- (z_s): NII shortfall in scenario (s).

The signed hedge is:

[
h = h^+ - h^-
]

Scenario NII is:

[
NII_s =
sum_{a in A} r^A_{sa}x_a
-
sum_{f in F} r^F_{sf}y_f
+
q_s(h^+-h^-)
]

The one-period objective is:

[
min
-sum_{s in S} p_s NII_s
+
lambda sum_{s in S}p_s z_s
+
c_h(h^+ + h^-)
]

subject to the balance-sheet, capital, liquidity, duration-gap, bucket-bound, hedge-bound, and NII-shortfall constraints described below.

## 2. Multi-period stochastic ALM

The stochastic model is represented on a scenario tree.

Let:

- (N): scenario-tree nodes;
- (t(n)): period of node (n);
- (p_n): probability of reaching node (n);
- (pi(n)): parent of non-root node (n);
- (x_{na}): allocation to asset (a) at node (n);
- (y_{nf}): funding from source (f) at node (n);
- (h_n^+,h_n^-): positive/negative hedge notionals;
- (z_n): NII shortfall;
- (u^A_{na},v^A_{na}): positive/negative asset rebalancing;
- (u^F_{nf},v^F_{nf}): positive/negative funding rebalancing.

Because decisions are indexed by scenario-tree nodes rather than complete future paths, non-anticipativity is implicit: before two paths diverge, they use the same node decision.

### Node NII

[
NII_n =
sum_{a in A} r^A_{na}x_{na}
-
sum_{f in F} r^F_{nf}y_{nf}
+
q_n(h_n^+-h_n^-)
]

### Expected objective

The stochastic program minimizes:

[
-sum_{n in N}p_n NII_n
+
lambdasum_{n in N}p_n z_n
+
c_Asum_{n 
eq root}p_n
sum_{a in A}(u^A_{na}+v^A_{na})
]

[
+
c_Fsum_{n 
eq root}p_n
sum_{f in F}(u^F_{nf}+v^F_{nf})
+
c_hsum_{n in N}p_n(h_n^+ + h_n^-)
]

The first term rewards expected NII. The remaining terms penalize downside, balance-sheet turnover, funding turnover, and gross hedge usage.

## 3. Balance-sheet identities

For every node:

[
sum_{a in A}x_{na}=B
]

[
sum_{f in F}y_{nf}=B
]

where (B) is the modeled balance-sheet size.

## 4. Rebalancing identities

For each non-root node (n):

[
x_{na}-x_{pi(n),a}
=
u^A_{na}-v^A_{na}
]

and:

[
y_{nf}-y_{pi(n),f}
=
u^F_{nf}-v^F_{nf}
]

with all turnover variables nonnegative.

These variables linearize absolute changes and allow transaction/rebalancing costs to be included in the objective.

## 5. Capital constraint

Let (E) be equity, (w_a) risk weights, and (kappa) the minimum capital ratio.

For every node:

[
sum_{a in A}w_a x_{na}
le
rac{E}{kappa}
]

This is an educational RWA-based capital proxy, not a regulatory capital engine.

## 6. Liquidity constraint

Let:

- (ell_a): liquid-asset weight;
- (ho_f): base runoff coefficient;
- (m_n): node-specific runoff stress multiplier.

For every node:

[
sum_{a in A}ell_a x_{na}
ge
sum_{f in F}m_nho_f y_{nf}
]

This creates state-dependent liquidity pressure.

## 7. Duration-gap constraint

Let (d_a) and (d_f) be duration proxies, (d_h) hedge duration sensitivity, and (Delta) the maximum absolute duration gap.

For every node:

[
-Delta B
le
sum_{a in A}d_a x_{na}
-
sum_{f in F}d_f y_{nf}
+
d_h(h_n^+-h_n^-)
le
Delta B
]

## 8. NII downside constraint

Let (T_t) be the NII target for period (t).

For every node:

[
z_n ge T_{t(n)} - NII_n
]

[
z_n ge 0
]

The objective penalizes expected downside rather than enforcing the NII target as a hard feasibility constraint.

## 9. Bounds

Asset and funding variables have explicit bucket-specific lower and upper bounds.

For the hedge:

[
0 le h_n^+,h_n^- le H_{max}
]

All turnover and shortfall variables are nonnegative.

## 10. Static-policy benchmark

For the non-adaptive benchmark, all future balance-sheet and signed-hedge decisions are frozen to the root decision.

For every non-root node:

[
x_{na}=x_{root,a}
]

[
y_{nf}=y_{root,f}
]

[
h_n^+ - h_n^-
=
h_{root}^+ - h_{root}^-
]

The static model therefore has the same scenario assumptions and risk constraints but cannot react to new information.

Since the adaptive model contains all feasible static policies plus additional recourse flexibility, its minimized objective cannot be worse than the static benchmark, up to numerical tolerance.

## 11. Value of adaptivity

The project reports:

[
VoA =
Objective_{static}
-
Objective_{adaptive}
]

Because the model is written as a minimization problem, a positive value indicates an improvement from adaptive recourse under the synthetic scenario tree.

This quantity is a model-comparison diagnostic, not an estimate of real-world financial value.
