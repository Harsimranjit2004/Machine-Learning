import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def handle_missing_values(df, strategy = 'mean', numerical_fill = None, categorical_fill = None):
    """
    Handle missing values with simple imputer

    Parameters:
    - df: dataframe
    - strategy: str, default='mean'
        Options: 'mean', 'median', 'most_frequent', 'constant'
    - numerical_fill_value: str or numerical value, default=None
        Used with strategy='constant' for numerical columns.
    - categorical_fill_value: str or categorical value, default=None
        Used with strategy='constant' for categorical columns.

    Returns:
    - df: pandas DataFrame
        DataFrame with missing values handled 
    """
    if strategy not in ['mean', 'median', 'most_frequent','constant' ]:
        raise ValueError("Invalid strategy. Options are 1. mean 2. median 3. most_frequent 4. constant")
    
    numerical_cols = df.select_dtypes(include= np.number).columns
    categorical_cols = df.select_dtypes(exclude = np.number).columns

    if strategy == 'constant' and numerical_fill is None:
        raise ValueError("For 'constant' strategy, numerical_fill_value must be specified.")
    
    if strategy == 'constant' and categorical_fill is None:
        raise ValueError("For 'constant' strategy, categorical_fill_value must be specified.")
    
    numerical_imputer = SimpleImputer(strategy=strategy, fill_value=numerical_fill)
    df[numerical_cols]  = numerical_imputer.fit_transform(df[numerical_cols])

    categorical_imputer = SimpleImputer(strategy=strategy, fill_value=categorical_fill)
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    return df
data = {
    'A': [1, 2, np.nan, 4],
    'B': ['a', np.nan, 'c', 'd'],
    'C': [5.0, np.nan, 7.0, 8.0],
    'D': [np.nan, 'x', 'y', np.nan]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Test the handle_missing_values function
filled_df = handle_missing_values(df, strategy='mean', numerical_fill=0, categorical_fill='missing')
print(filled_df)