#Handles non-intensive bootstrapping jobs
import statsmodels.api as sm
import numpy as np

def bootstrap_coefficient(df, dependent_variable, main_independent_variable, controls, iterations):
    '''Bootstrap the main coefficient from the full model'''
    x_columns = [main_independent_variable] + controls
    coefficients = []
    for _ in range(iterations):
        sample_df = df.sample(
            n=len(df),
            replace=True
        )

        y = sample_df[dependent_variable]
        X = sample_df[x_columns]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        coefficients.append(float(model.params[main_independent_variable]))

    coefficients = np.array(coefficients)

    return {
        "mean": float(np.mean(coefficients)),
        "standard_error": float(np.std(coefficients, ddof=1)),
        "ci_95": [
            float(np.percentile(coefficients, 2.5)),
            float(np.percentile(coefficients, 97.5)),
        ],
        "samples": coefficients.tolist(),
    }