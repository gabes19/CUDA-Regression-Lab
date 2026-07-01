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
    #TODO: Temporarily returning something simple. Expand after cupy math
    return {
         "row_count": len(df),
         "columns": required_columns,
         "x_shape": X.shape,
         "y_shape": y.shape
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
     y = df[dependent_variable].astype("float").to_cupy()
     X = df[x_columns].astype("float64").to_cupy()

     n_rows = X.shape[0]
     intercept = cp.ones((n_rows, 1), dtype=cp.float64)
     X = cp.column_stack([intercept, X])

     return X, y



def _fit_ols_gpu(X, y):
    pass


