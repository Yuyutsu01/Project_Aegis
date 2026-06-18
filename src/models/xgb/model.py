import xgboost as xgb
import pandas as pd
import numpy as np

class XGBDirectionalModel:
    def __init__(self):
        # Increased regularization and constrained depth to prevent extreme overfitting
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=50
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
