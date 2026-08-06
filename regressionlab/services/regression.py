#This service handles model fitting
import numpy as np
import statsmodels.api as sm

def clean_metric(value):
    '''Return JSON/template-friendly floats for model metrics.'''
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(metric):
        return None

    return metric

def fit_models(df, dependent_variable, main_independent_variable, controls):
    '''Fit multiple models and return results'''
    y = df[dependent_variable]
    model_results = []
    for i in range(len(controls) +1):
        current_controls = controls[:i]
        x_columns = [main_independent_variable] + current_controls
        
        X = df[x_columns]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        coefficient_interval = model.conf_int().loc[main_independent_variable]

        model_results.append({
            "model_name": f"Model {i+1}",
            "formula": f"{dependent_variable} ~ " + " + ".join(x_columns),
            "controls": current_controls,
            "coefficient": clean_metric(model.params[main_independent_variable]),
            "standard_error": clean_metric(model.bse[main_independent_variable]),
            "t_value": clean_metric(model.tvalues[main_independent_variable]),
            "p_value": clean_metric(model.pvalues[main_independent_variable]),
            "ci_95": [
                clean_metric(coefficient_interval[0]),
                clean_metric(coefficient_interval[1]),
            ],
            "r_squared": clean_metric(model.rsquared),
            "adjusted_r_squared": clean_metric(model.rsquared_adj),
            "rmse": clean_metric(np.sqrt(model.mse_resid)),
            "f_statistic": clean_metric(model.fvalue),
            "f_p_value": clean_metric(model.f_pvalue),
            "n_observations": int(model.nobs),
            "df_residual": clean_metric(model.df_resid),
            "condition_number": clean_metric(model.condition_number),
        })

    return model_results