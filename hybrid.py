import numpy as np
from xgboost import XGBClassifier


def train_hybrid(X_train, X_test, y_train, fuzzy_simulator, fuzzy_func):

    fuzzy_train = np.array([fuzzy_func(fuzzy_simulator, row) for row in X_train])
    fuzzy_test = np.array([fuzzy_func(fuzzy_simulator, row) for row in X_test])

    X_train_h = np.column_stack((X_train, fuzzy_train))
    X_test_h = np.column_stack((X_test, fuzzy_test))

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train_h, y_train)
    y_pred = model.predict(X_test_h)

    return y_pred