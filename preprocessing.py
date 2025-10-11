# preprocessing.py
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Удаляем псевдопропуски
    cat_columns = df.select_dtypes(exclude='number').columns
    if len(cat_columns) > 0:
        mask = df[cat_columns].isin(['', '-', '–', '—', 'NaN', 'None'])
        to_drop = mask.any(axis=1)
        df = df.loc[~to_drop]

    # Устраняем пропуски
    df = df.dropna()

    # Устраняем полные явные дубликаты
    df = df.drop_duplicates()

    # Перенумерация индекса
    df = df.reset_index(drop=True)

    return df


def iqr_remove_outlier_rows(df: pd.DataFrame, columns=None, factor: float = 4) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    if not columns:
        return df.reset_index(drop=True)

    for col in columns:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        df = df.loc[(s >= low) & (s <= high)]

    return df.reset_index(drop=True)


# Функция для распределения признаков на категориальные и количественные
def distribution_by_data_types(df: pd.DataFrame):
    quantitative_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns  = df.select_dtypes(exclude='number').columns.tolist()
    return quantitative_columns, categorical_columns


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    num_columns, ord_columns = distribution_by_data_types(X)

    ord_pipe = Pipeline([
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    data_preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ord_pipe, ord_columns),
            ('num', StandardScaler(), num_columns)
        ],
        remainder='passthrough'
    )

    X_train = data_preprocessor.fit_transform(X)
    feature_names = data_preprocessor.get_feature_names_out()
    X_final = pd.DataFrame(X_train, columns=feature_names, index=df.index)

    return X_final

