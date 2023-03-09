# %%
from __future__ import division
import os
import datetime
import random
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import torch

import optuna

from libs import CreateData
from libs.NeuralBW import train_and_enhance_NBW
from libs.NeuralBW import estimate_balancing_weights
from libs.OptunaFuncitons import objective_GBT

# Script Global Parameters
EXPERIMENT_TYPE = 'Experiment1'

"""
Setting for run script
"""
# For neural network structure
HIDDEN_DIM_NBW = 20
N_LAYERS_NBW = 8

# For mini-batch stochastic gradient optimization in training:
LEARNING_RATE = 0.0001
BATCHSIZE = 2500
EPOCHS_NBW = 1000

# For estimateing the balancing weights:
ALPHA = 1/2
N_ENHANCE_NBW_MODELS = 5

# For output directory
OUTPUT_DIRECTORY = './out'

# For Generating data
N_DATA_TEST = 10000
N_DATA_TRAIN = 10000
N_DATA_ALL = N_DATA_TRAIN + N_DATA_TEST

# For hyperparameter searches for GDB
N_TRY_OPT = 25
RATE_SUBSAMPLE_TUNE_GDB = 0.2
OPTUNA_TIMEOUT_SECONDS = 60*(60*6)


# %%
np.random.seed(1)
random.seed(1)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
if __name__ == '__main__':
    now = datetime.datetime.now()
    current_datetime = now.strftime('%Y%m%d_%H%M_%S')

    """
    Generate data
    """
    train_expls, train_resp, test_expls, test_resp, \
        explanatories_intervention, true_response_intervention\
        = CreateData.gen(
            N_DATA_TRAIN,
            N_DATA_TEST,
            EXPERIMENT_TYPE)
    train_expls_mat = np.concatenate(train_expls, axis=1)
    test_expls_mat = np.concatenate(test_expls, axis=1)
    expls_intervention_mat = np.concatenate(
        explanatories_intervention, axis=1)
    expls_refit_mat = np.concatenate(
        [train_expls_mat, test_expls_mat], axis=0)
    resp_refit = np.concatenate(
        [train_resp, test_resp], axis=0)
    expls_refit_tsr = torch.from_numpy(
        expls_refit_mat)

    """
    Run a experment
    """
    print(
      f'Start Building a NGB model... --- current time: {current_datetime}')

    # Structure of a NGB model
    params_nbw = {
        'n_layers': N_LAYERS_NBW,
        'hidden_dim': HIDDEN_DIM_NBW}

    # Directory to output results and logs
    out_parent_dir = os.path.join(
        OUTPUT_DIRECTORY,
        f'{EXPERIMENT_TYPE}_{current_datetime}')
    out_result_dir = os.path.join(out_parent_dir, 'all_results')

    # Build nbw models while trying to enhance the balancing of
    # the weights (N_ENHANCE_NBW_MODELS times)
    filePaths_of_nbw_models_to_use_list, \
        final_alpha_infomation_estimated, \
        alpha_infos_of_all_nbw_models = train_and_enhance_NBW(
            ALPHA,
            train_expls,
            params_nbw,
            N_ENHANCE_NBW_MODELS,
            LEARNING_RATE,
            BATCHSIZE,
            EPOCHS_NBW,
            out_result_dir)

    print('--------------------------------------------------------------')
    print(f'NBW Models to use: {filePaths_of_nbw_models_to_use_list}')
    print(f'Estimateid alpha-information for the balanced distribution = ',
          f'{final_alpha_infomation_estimated}')
    print('--------------------------------------------------------------')

    # Estimate the balancig weights
    balancing_weights_estimated_train = estimate_balancing_weights(
        train_expls,
        params_nbw,
        filePaths_of_nbw_models_to_use_list)
    balancing_weights_estimated_test = estimate_balancing_weights(
        test_expls,
        params_nbw,
        filePaths_of_nbw_models_to_use_list)
    balancing_weights_estimated_refit = np.concatenate(
       [balancing_weights_estimated_train,
        balancing_weights_estimated_test], axis=0)

    """
    Search hyperparameters for Gradient Boosting Tree(GBT) with the
    balancig weights
    """
    study_nbw = optuna.create_study(direction='minimize')
    study_nbw.optimize(
        objective_GBT(
          train_expls_mat,
          train_resp,
          test_expls_mat,
          test_resp,
          RATE_SUBSAMPLE_TUNE_GDB,
          balancing_weights_estimated_train
          ),
        n_trials=N_TRY_OPT,
        timeout=OPTUNA_TIMEOUT_SECONDS
    )
    best_trial_nbw = study_nbw.best_trial
    print(best_trial_nbw)

    # Built a model of GBT
    learning_rate_nbw = best_trial_nbw.params['learning_rate']
    max_leaf_nodes_nbw = best_trial_nbw.params['max_leaf_nodes']
    model_of_GBT = GradientBoostingRegressor(
            random_state=0,
            max_leaf_nodes=max_leaf_nodes_nbw,
            learning_rate=learning_rate_nbw)
    model_of_GBT.fit(
        expls_refit_mat, resp_refit,
        sample_weight=balancing_weights_estimated_refit)

    # Estimate the averate causal effect frum the model of GBT
    Y_estimated_GBT_nbw = model_of_GBT.predict(
        expls_intervention_mat)

    # Mean square errors
    mse_GBT_nbw = mean_squared_error(
      true_response_intervention,
      Y_estimated_GBT_nbw)

    print('--------------------------------------------------------------')
    print(f'Mean Square Error of Estimation = {mse_GBT_nbw}')
    print('--------------------------------------------------------------')

    # Save the above result (the model of GBT)
    basename_output = 'model_of_GBT'
    result_to_output = model_of_GBT
    with open(
        os.path.join(
            out_result_dir, basename_output + '.pickle'), 'wb') as wf:
        pickle.dump(result_to_output, wf)
