import numpy as np
from sklearn.metrics import make_scorer

def business_cost(y_true, y_pred_proba, threshold=0.5, fn_cost=10, fp_cost=1, **kwargs):
    y_pred = (y_pred_proba >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return -(fn * fn_cost + fp * fp_cost)

business_scorer = make_scorer(business_cost, needs_proba=True, greater_is_better=False)