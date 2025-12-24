import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

class XGBDirectionalModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
