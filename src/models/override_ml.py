# src/models/wrappers.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

class BaseModelWrapper:
    """
    全てのモデルラッパーの基底クラス
    run_ml.pyからはこのインターフェースを通じて呼び出される
    """
    def fit_predict(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

class LassoWrapper(BaseModelWrapper):
    def __init__(self, alpha=1.0, normalization="standard", random_state=42, **kwargs):
        self.alpha = alpha
        self.normalization = normalization
        self.random_state = random_state
        self.model = None

    def fit_predict(self, X_train, y_train, X_val, y_val):
        """
        学習を行い、Validationデータに対する予測値と学習済みモデルを返す
        """
        # 1. モデルの構築 (Pipeline)
        base_model = Lasso(alpha=self.alpha, random_state=self.random_state)
        
        # 正規化の切り替えロジックをここに集約
        if self.normalization == "standard":
            self.model = make_pipeline(StandardScaler(), base_model)
        elif self.normalization == "l2":
            self.model = make_pipeline(Normalizer(norm='l2'), base_model)
        else:
            self.model = base_model


        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_val).flatten()
        
        return preds, self

    def predict(self, X_test):
        """
        テストデータの予測
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        return self.model.predict(X_test).flatten()

class LGBMWrapper(BaseModelWrapper):
    def __init__(self, params=None, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100, **kwargs):
        """
        params: lightgbm.yaml から渡されるハイパーパラメータ辞書
        """
        self.params = params if params is not None else {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model = None
        self.encoders = {} # カテゴリ変数のエンコーダー保存用


    def fit_predict(self, X_train, y_train, X_val, y_val):
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        # 4. 学習
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(self.verbose_eval)
            ]
        )
        
        preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        
        return preds, self

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)
