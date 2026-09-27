# Mathematical Model

Let:

- (A): asset buckets;
- (F): funding sources;
- (S): scenarios;
- (x_a): allocation to asset (a);
- (y_f): amount raised from funding source (f);
- (h^+, h^-): positive and negative hedge notionals;
- (z_s): NII shortfall in scenario (s).

The signed hedge is (h=h^+-h^-).

## Scenario NII

For scenario (s):

[
NII_s =
sum_{a in A} r^A_{sa} x_a
-
sum_{f in F} r^F_{sf} y_f
+
q_s(h^+-h^-)
]

where (r^A) are asset yields, (r^F) are funding costs, and (q_s) is the per-unit hedge payoff.

## Objective

[
min
-
sum_s p_s NII_s
+
lambda sum_s p_s z_s
+
c_h(h^+ + h^-)
]

The first term rewards expected NII. The second penalizes downside relative to the scenario target. The third discourages unnecessary gross hedge positions.

## Balance-sheet identities

[
sum_a x_a = B
]

[
sum_f y_f = B
]

## Capital proxy

With equity (E), risk weights (w_a), and minimum capital ratio (kappa):

[
sum_a w_a x_a le rac{E}{kappa}
]

## Liquidity proxy

Let (ell_a) be a liquid-asset weight and (ho_f) a stressed funding runoff rate:

[
sum_a ell_a x_a
ge
sum_f ho_f y_f
]

## Duration-gap limit

Let (d_a) and (d_f) be durations, (d_h) hedge duration sensitivity, and (Delta) the maximum permitted gap:

[
-Delta B
le
sum_a d_a x_a
-
sum_f d_f y_f
+
d_h(h^+-h^-)
le
Delta B
]

## Scenario shortfall

For NII target (T):

[
z_s ge T - NII_s,qquad z_s ge 0
]

## Bounds

All asset and funding buckets have explicit lower/upper bounds, and:

[
0 le h^+,h^- le H_{max}
]

The complete model is a linear program.
