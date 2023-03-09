from typing import List, Any, Tuple, Dict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import optuna


def objective_GBT(
      X_train: Any,
      y_train: Any,
      X_test: Any,
      y_test: Any,
      rate_subsample_tune_gdb,
      sample_weight_train=None,
      ) -> float:
    def _objective_gbt(trial: optuna.trial.Trial):
        N_data = X_train.shape[0]
        max_leaf_nodes = trial.suggest_int(
            'max_leaf_nodes', 2, int(N_data/2), log=True)
        learning_rate = trial.suggest_float('learning_rate', 0.0, 0.5)
        reg = GradientBoostingRegressor(
                random_state=0,
                max_leaf_nodes=max_leaf_nodes,
                subsample=rate_subsample_tune_gdb,
                learning_rate=learning_rate)
        if sample_weight_train is not None:
            reg.fit(X_train, y_train, sample_weight_train)
            y_estimated = reg.predict(X_test)
            mse = mean_squared_error(y_estimated, y_test)
        else:
            reg.fit(X_train, y_train)
            y_estimated = reg.predict(X_test)
            mse = mean_squared_error(y_estimated, y_test)
        return mse
    return _objective_gbt

