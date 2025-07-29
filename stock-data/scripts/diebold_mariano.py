import numpy as np
from statsmodels.regression.linear_model import OLS
from scipy.stats import norm

def diebold_mariano_test(
    y_true1, y_pred1,
    y_true2, y_pred2,
    h=1,
    loss='mse'
):
    """
    Diebold–Mariano test comparing losses on two separate series.

    Parameters
    ----------
    y_true1 : array-like
        Ground truth for series 1.
    y_pred1 : array-like
        Forecasts for series 1.
    y_true2 : array-like
        Ground truth for series 2.
    y_pred2 : array-like
        Forecasts for series 2.
    h       : int
        Forecast horizon (used to set HAC lag = h-1).
    loss    : {'mse','mae'}
        Which loss to apply.

    Returns
    -------
    dm_stat : float
        t–statistic testing mean loss1 − mean loss2 ≠ 0.
    p_value : float
        two‐sided p‐value.
    """
    # Convert inputs
    y1 = np.asarray(y_true1)
    f1 = np.asarray(y_pred1)
    y2 = np.asarray(y_true2)
    f2 = np.asarray(y_pred2)

    if not (len(y1)==len(f1)==len(y2)==len(f2)):
        raise ValueError("All inputs must have the same length")

    # 1) Compute loss series
    if loss == 'mse':
        L1 = (f1 - y1)**2
        L2 = (f2 - y2)**2
    elif loss == 'mae':
        L1 = np.abs(f1 - y1)
        L2 = np.abs(f2 - y2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    # 2) Loss differential d_t
    d = L1 - L2
    T = len(d)

    # 3) Regress d on constant, use HAC for serial correlation
    X = np.ones((T,1))
    model = OLS(d, X).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': h-1}
    )

    # model.params[0] is mean(d), model.bse[0] is its se
    mean_diff = model.params[0]
    se_diff   = model.bse[0]

    dm_stat = mean_diff / se_diff
    p_value = 2 * norm.sf(abs(dm_stat))

    return dm_stat, p_value
