from fluid_benchmarking import estimators, irt_utils 
import numpy as np

theta_anchor = theta_of(y_anchor, a_anchor, b_anchor)
resid = residual(y_held, theta_anchor, a_held, b_held)

# Can write this today
d1 = roc_auc_score(is_contam, resid)                                    # + bootstrap CI


d2 = theta_of(y_contam, a_contam, b_contam) - theta_of(y_clean, a_clean, b_clean)
d3 = resid[is_contam].mean() - resid[~is_contam].mean()