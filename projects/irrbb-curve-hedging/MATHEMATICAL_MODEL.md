# Mathematical Model

Let k index key-rate tenors, j candidate hedge instruments and s curve-shock
scenarios.

The unhedged banking-book key-rate PV01 is b_k. Hedge j contributes h_kj PV01
per unit of signed notional x_j.

Residual curve exposure is:

~~~text
r_k = b_k + sum_j h_kj x_j
~~~

For scenario s with tenor shock q_sk in basis points, the first-order EVE
change is:

~~~text
DeltaEVE_s = - sum_k q_sk r_k
~~~

so the positive loss proxy is:

~~~text
L_s = sum_k q_sk r_k
~~~

Introduce z as the worst scenario loss and split signed hedge notional into:

~~~text
x_j = x_j^+ - x_j^-
x_j^+, x_j^- >= 0
~~~

The optimization is:

~~~text
minimize
    z + lambda sum_j c_j (x_j^+ + x_j^-)

subject to
    sum_k q_sk r_k <= z          for every scenario s

    sum_j (x_j^+ + x_j^-) <= G

    0 <= x_j^+ <= U_j
    0 <= x_j^- <= U_j
    z >= 0
~~~

where c_j is hedge cost, G is the gross hedge budget and U_j is the
instrument-level bound.

The formulation is a linear program solved with SciPy/HiGHS.

The stress set and PV01 values are synthetic and are not presented as a
regulatory calibration.
