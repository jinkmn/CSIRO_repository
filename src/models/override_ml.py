# src/models/wrappers.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Lasso, Ridge # Ridge追加
from sklearn.ensemble import GradientBoostingRegressor
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
        base_model = Lasso(alpha=self.alpha, random_state=self.random_state) 
        
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
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X_test).flatten()


class RidgeWrapper(BaseModelWrapper):
    def __init__(self, alpha=1.0, normalization="standard", random_state=42, **kwargs):
        self.alpha = alpha
        self.normalization = normalization
        self.random_state = random_state
        self.model = None

    def fit_predict(self, X_train, y_train, X_val, y_val):
        # Ridgeにもrandom_stateを渡す (solver='sag'などの場合に影響するため推奨)
        base_model = Ridge(alpha=self.alpha, random_state=self.random_state)
        
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
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X_test).flatten()

class GBRWrapper(BaseModelWrapper):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, normalization="standard", random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.normalization = normalization
        self.random_state = random_state
        self.kwargs = kwargs # その他のパラメータ
        self.model = None

    def fit_predict(self, X_train, y_train, X_val, y_val):
        # random_state を指定
        base_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.kwargs
        )
        
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
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X_test).flatten()


class LGBMWrapper(BaseModelWrapper):
    def __init__(self, params=None, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100, random_state=42, **kwargs):
        self.params = params if params is not None else {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.random_state = random_state # 保存
        self.model = None
        
        self.params['random_state'] = self.random_state
        self.params['bagging_seed'] = self.random_state
        self.params['feature_fraction_seed'] = self.random_state

    def fit_predict(self, X_train, y_train, X_val, y_val):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
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

class CatBoostWrapper(BaseModelWrapper):
    def __init__(self, params=None, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100, random_state=42, **kwargs):
        self.params = params if params is not None else {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.random_state = random_state
        self.model = None

        self.params['random_seed'] = self.random_state

    def fit_predict(self, X_train, y_train, X_val, y_val):
        # CatBoost用 Pool作成
        train_pool = cb.Pool(X_train, y_train)
        val_pool = cb.Pool(X_val, y_val)

        self.model = cb.train(
            params=self.params,
            dtrain=train_pool,
            num_boost_round=self.num_boost_round,
            evals=[val_pool],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval
        )
        
        # CatBoostのpredictはデフォルトでbest_iterationを使うが、明示的に指定も可能
        preds = self.model.predict(val_pool)
        return preds, self

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        test_pool = cb.Pool(X_test)
        return self.model.predict(test_pool)