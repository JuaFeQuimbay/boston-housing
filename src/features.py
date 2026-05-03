# src/features.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_feature_names(X):
    """
    Retorna las columnas numéricas (Boston Housing es 100% numérico)
    """
    return X.columns.tolist()


def build_preprocessing_pipeline(X):
    """
    Construye el pipeline de preprocesamiento
    """

    numeric_features = get_feature_names(X)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features)
        ]
    )

    return preprocessor