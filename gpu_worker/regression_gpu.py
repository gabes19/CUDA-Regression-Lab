#This module runs in the Runpod cloud-hosted 
#Linux GPU worker environment.
import time
import cudf
import cupy as cp
from scipy import stats

def run_gpu_analysis(
        csv_path,
        dependent_variable,
        main_independent_variable,
        controls,
        bootstrap_iterations = 500,
        bootstrap_indices = None
):
    required_columns = [dependent_variable, main_independent_variable] + controls
    df = _load_numeric_gpu_frame(csv_path = csv_path, required_columns=required_columns)
    x_columns = [main_independent_variable]
    X, y = _build_design_matrices(
        df=df,
        dependent_variable=dependent_variable,
        x_columns=x_columns,
        )
    ols_result = _fit_ols_gpu(X, y)
    #TODO: Temporarily returning something simple. Expand after cupy math
    return {
         "row_count": len(df),
         "columns": required_columns,
         "x_shape": X.shape,
         "y_shape": y.shape,
         "intercept": float(ols_result["beta"][0].get()),
         "coefficient": float(ols_result["beta"][1].get())
    }
    
#TODO: add support for cpu/gpu categorical variables later
def _load_numeric_gpu_frame(csv_path, required_columns):
    '''Loads the data in a cudf and keeps only required columns'''
    df = cudf.read_csv(csv_path)
    df = df[required_columns].dropna()
    for column in required_columns:
        if not cudf.api.types.is_numeric_dtype(df[column].dtype):
                raise ValueError(
                        f"GPU regression currently supports numeric columns only. "
                        f"Column '{column}' has dtype {df[column].dtype}."
                        )
    return df

def _build_design_matrices(df, dependent_variable, x_columns):
    y = df[dependent_variable].astype("float64").to_cupy()
    X = df[x_columns].astype("float64").to_cupy()

    n_rows = X.shape[0]
    intercept = cp.ones((n_rows, 1), dtype=cp.float64)
    X = cp.column_stack([intercept, X])

    return X, y

def _fit_ols_gpu(X,y):
    '''Fit model using Moore-Penrose pseudoinverse of X 
    (generalization of the inverse matrix with 
    Singular Value Decomposition (SVD))'''
    #pseudo inverse of X
    x_pinv = cp.linalg.pinv(X)
    #matmul computes estimated coefficients
    beta = x_pinv @ y
    #matmul computes predicted y vals
    fitted_values = X @ beta
    #computes errors
    residuals = y -fitted_values
    n_observations = X.shape[0]
    n_parameters = X.shape[1]
    #residual degrees of freedom
    df_residual = n_observations - n_parameters
    #sum of squared errors
    sse = cp.sum(residuals ** 2)
    #total sum of squares
    tss = cp.sum((y - cp.mean(y)) ** 2)
    #r^2 - how much variation is explained by model
    r_squared = 1 - (sse / tss)
    #residual mean squared error - used to compute standard error
    mse_resid = sse / df_residual
    #root mean squared error - puts error in original y units
    rmse = cp.sqrt(mse_resid)



    return {
        "beta": beta,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "sse": sse,
        "tss": tss,
        "r_squared": r_squared,
        "mse_resid": mse_resid,
        "rmse": rmse,
        "n_observations": n_observations,
        "df_residual": df_residual,
    }


