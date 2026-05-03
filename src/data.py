# src/data.py

import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    """
    Carga el dataset Boston Housing desde OpenML
    """
    data = fetch_openml(name="boston", version=1, as_frame=True)
    df = data.frame

    return df


def split_data(df):
    """
    Separa features y target
    """
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]

    return X, y