import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)

    df.fillna(method='ffill', inplace=True)

    # Save encoders for consistency
    encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_column = df.columns[-1]

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add slight noise
    X_train += np.random.normal(0, 0.1, X_train.shape)

    return X_train, X_test, y_train, y_test, scaler, encoders, X.columns


def train_models(X_train, y_train):

    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    cat = CatBoostClassifier(
        iterations=50,
        depth=4,
        learning_rate=0.1,
        verbose=0,
        random_seed=42
    )

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    return xgb, cat