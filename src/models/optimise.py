from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import sklearn.metrics as metrics

metrics_regression = {
    'explained_variance' : metrics.explained_variance_score,
    'max_error' : metrics.max_error,
    'neg_mean_absolute_error' : metrics.mean_absolute_error,
    'neg_mean_squared_error' : metrics.mean_squared_error,
    'neg_root_mean_squared_error' : metrics.mean_squared_error,
    'neg_mean_squared_log_error' : metrics.mean_squared_log_error,
    'neg_median_absolute_error' : metrics.median_absolute_error,
    'r2' : metrics.r2_score,
    'neg_mean_poisson_deviance' : metrics.mean_poisson_deviance,
    'neg_mean_gamma_deviance' : metrics.mean_gamma_deviance
}


def lightgbm_cv(params, data, targets, scoring='neg_mean_squared_error', cv=10):
    """
    LightGBM Cross Validation.
    This function will instantiate a LGBMRegressor with parameters
    @params. Combined with data and targets this will in turn be used to 
    perform cross validation. The result of the @scoring using
    cross-validation with @cv folds is returned.
    

    Parameters
    ----------
    params: dictionary with the LGBMRegressor Params to optmize
    data: matrix with variable values for training
    targets: vector with the target values for training 
    scoring='neg_mean_squared_error': metric to use to calculate the model performance
    cv=10: number of folds to run the cross-validation

    Returns
    -------
    cross-validation mean
    """
    estimator = LGBMRegressor(**params)
    cval = cross_val_score(estimator, data, targets, scoring=scoring, cv=cv)
    return cval.mean()


def xgboost_cv(params, data, targets, scoring='neg_mean_squared_error', cv=10):
    """
    Xgboost Regressor Cross Validation.
    This function will instantiate a XGBRegressor with parameters
    @params. Combined with data and targets this will in turn be used to 
    perform cross validation. The result of the @scoring using
    cross-validation with @cv folds is returned.
    

    Parameters
    ----------
    params: dictionary with the XGBRegressor Params to optmize
    data: matrix with variable values for training
    targets: vector with the target values for training 
    scoring='neg_mean_squared_error': metric to use to calculate the model performance
    cv=10: number of folds to run the cross-validation

    Returns
    -------
    cross-validation mean
    """
    estimator = XGBRegressor(**params)
    cval = cross_val_score(estimator, data, targets, scoring=scoring, cv=cv)
    return cval.mean()


def optimise_lightgbm(data, targets):
    pass

def optimise_xgb(data, targets):
    pass

