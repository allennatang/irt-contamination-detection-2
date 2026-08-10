# 08/08/2026
Wrote irt.py.

# 08/10/2026
Tier 0 regression passes: `tests/test_regression_v1.py`, 5 tests, CPU, <1s.

    .venv/bin/python -m pytest tests/test_regression_v1.py -v

Recomputed theta + residuals from the correctness column of v1's
`results/mink_plus_plus/irt_mink_join_orig_loglik.csv` against the prefit bank
(`allenai/fluid-benchmarking`, `data/irt_models/mmlu.csv`), n=1215 items
(v1's 1,216 minus one duplicate question stem).

| model | n | theta (MAP) | SE(theta) | max abs residual diff vs v1 |
|---|---|---|---|---|
| base | 1215 | -0.500973 | 0.0895 | 1.11e-16 |
| orig_adapter | 1215 | +0.468675 | 0.1127 | 1.11e-16 |

Match is at float epsilon, not the 1e-6 the guide asked for. IRT layer is correct.
Note the fixture ships no `a`/`b` columns despite the guide's claim -- they come
from the bank, joined on item_id.