#Handles parsing and data processing for input
import pandas as pd



#Define maximum amount of categories for one-hot encoding
MAX_AUTO_CATEGORIES = 25


def is_categorical_series(series: pd.Series) -> bool:
    '''Helper function that checks if a series is categorical
    for later encoding'''
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )

def get_category_levels(series: pd.Series) -> list:
    '''Helper function that returns a list of category labels sorted by frequency desc'''
    counts = series.dropna().astype(str).value_counts()
    return sorted(
        counts.index.tolist(), 
        key= lambda level: (-int(counts[level]), level.casefold()),
    )

def parse_columns(csv_path):
    '''Helper function to parse column metadata for
    user to choose target variables and prepare for LLM'''

    df = pd.read_csv(csv_path)
    column_metadata = []

    for column in df.columns:
        series = df[column]
        is_categorical = is_categorical_series(series)

        category_levels = (
            get_category_levels(series)
            if is_categorical
            else[]
        )

        unique_values = int(series.nunique(dropna=True))
        auto_encodable = (
            is_categorical
            and 2 <= unique_values <= MAX_AUTO_CATEGORIES
        )
        column_metadata.append({
            "name": column,
            "dtype": str(series.dtype),
            "semantic_type": (
                "categorical"
                if is_categorical
                else "numeric"
                if pd.api.types.is_numeric_dtype(series.dtype)
                else "unsupported"
            ),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique()),
            "auto_encodable": auto_encodable,
            "reference_level": (
                category_levels[0]
                if auto_encodable
                else None
            ),
            "encoded_column_count":(
                unique_values - 1
                if auto_encodable
                else 0
        ),
        })
    return column_metadata