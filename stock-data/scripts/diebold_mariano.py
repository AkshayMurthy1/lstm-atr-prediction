import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS
from scipy.stats import norm

def diebold_mariano_test(y_true, forecast1, forecast2, h=1, loss='mse'):
    """
    Diebold-Mariano test for equal forecast accuracy.
    
    Parameters:
        y_true     : array-like of true values
        forecast1  : predictions from model 1
        forecast2  : predictions from model 2
        h          : forecast horizon (default: 1)
        loss       : 'mse' or 'mae'

    Returns:
        DM statistic and p-value
    """

    # Compute loss differential series
    if loss == 'mse':
        d = (forecast1 - y_true) ** 2 - (forecast2 - y_true) ** 2
    elif loss == 'mae':
        d = np.abs(forecast1 - y_true) - np.abs(forecast2 - y_true)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    d = d - np.mean(d)  # mean-centered

    T = len(d)

    # Regress d_t on constant and get HAC variance
    X = np.ones((T, 1))
    model = OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags': h - 1})

    dm_stat = model.tvalues[0]
    p_value = 2 * norm.sf(np.abs(dm_stat))

    return dm_stat, p_value
