# %%
from __future__ import division
import os
import datetime
import random
import re
import pickle
import argparse

from typing import List, Any, Tuple
from typing import Optional

import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

import torch
from torch import nn, Tensor
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import optuna


def _calc_stas(vals_col):
    tmp = vals_col[vals_col <= 5]
    val_mean = np.mean(tmp)
    val_std = np.std(tmp)
    return val_mean, val_std


# Script Global Parameters
Log_dir_headname = 'Experiment2'
N_DATA_TEST = 10000
N_DATA_TRAIN = 10000
N_DATA_ALL = N_DATA_TRAIN + N_DATA_TEST

#  Number of hyperparameter searches for GDB
N_TRY_OPT = 25

# Setting for hyperparameter searches of GDB
RATE_SUBSUMPLE_TUNE_GDB = 0.2
OPTUNA_TIMEOUT_SECONDS = 60*(60*6)

# Setting for estimating of Entropy Balancing method
MAX_DEGREE_OF_POLY = 5


parser = argparse.ArgumentParser()

# 3. parser.add_argument
parser.add_argument('N', type=int, help='Number of simulations')
parser.add_argument('-output_dir',
                    help='The directory path to output results', default='./')
parser.add_argument('-alpha', type=float, default=0.5,
                    help='alpha-divegence to use for estimating desity')
parser.add_argument('-n_layer_nn', type=int, default=8,
                    help='aNumber of layers of a neural network')
parser.add_argument('-n_units_hidden', type=int, default=20,
                    help='Number of units of hidden layer')
parser.add_argument('-n_enhance_nbw', type=int, default=5,
                    choices=range(2, 10),
                    help='Number of times to enhance NBW')
parser.add_argument('-n_epochs', type=int, default=1000,
                    help='Number of Epochs')
parser.add_argument('-batchsize', type=int, default=2500,
                    help='Sample size of mini-batch')
parser.add_argument('-learning_rate', type=float, default=0.0001,
                    help='Learning rate for training neural networks')

args = parser.parse_args()

print('----------------------------------------------------------------------')
print(f'The script arguments are')
print(f'  - Number of simulations: {args.N}')
print(f'  - output_dir: {args.output_dir}')
print(f'  - alpha: {args.alpha}')
print(f'  - n_layer_nn: {args.n_layer_nn}')
print(f'  - n_units_hidden: {args.n_units_hidden}')
print(f'  - n_enhance_nbw: {args.n_enhance_nbw}')
print(f'  - n_epochs: {args.n_epochs}')
print(f'  - batchsize: {args.batchsize}')
print(f'  - learning_rate: {args.learning_rate}')
print('----------------------------------------------------------------------')


# read script argments
N_SIMULATION = args.N
ALPHA = args.alpha
N_LAYERS_DIV = args.n_layer_nn
HIDDEN_DIM_DIV = args.n_units_hidden
EPOCHS_DIV = args.n_epochs
N_DIV_MODELS = args.n_enhance_nbw
DIVN_INITNAIL_DIV = args.learning_rate
LEARNING_RATE = args.learning_rate
BATCHSIZE = args.batchsize
out_parent_dir = args.output_dir


out_parent_dir = os.path.join(out_parent_dir, Log_dir_headname)
os.makedirs(out_parent_dir, exist_ok=True)
out_result_dir = os.path.join(out_parent_dir, 'all_results')
os.makedirs(out_result_dir, exist_ok=True)

np.random.seed(1)
random.seed(1)

this_scirpt_name = os.path.splitext(
  os.path.split(__file__)[1])[0]
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


"""
Definition of Base Classes
"""
p = ALPHA
q= (1 - ALPHA)
class PrimitiveNet(nn.Module):
    def __init__(self,
                 dropout: float,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 response_index: int,
                 explanatory_index: int
                 ):
        super().__init__()
        layers: List[nn.Module] = []
        self.response_index_ = response_index
        self.explanatory_index_ = explanatory_index
        _h_input_dim: int = input_dim
        for _h_out_dim in hidden_dims:
            layers.append(nn.Linear(_h_input_dim, _h_out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            _h_input_dim = _h_out_dim
        layers.append(nn.Linear(_h_input_dim, output_dim))
        self.layers_: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layers_(data)


class DivNetList(nn.Module):
    def __init__(self,
                 params_for_primitiv_model: dict):
        super().__init__()
        self.dropout_ = params_for_primitiv_model['dropout']
        self.input_dim_ = params_for_primitiv_model['input_dim']
        self.hidden_dim_ = params_for_primitiv_model['hidden_dim']
        self.output_dim_ = params_for_primitiv_model['output_dim']
        self.n_layers_ = params_for_primitiv_model['n_layers']
        self.hidden_dims_ = [self.hidden_dim_] * (self.n_layers_ - 1)
        self.div_models_ = nn.ModuleList()

    def add_new_div_model(self):
        mdl = PrimitiveNet(
                            self.dropout_,
                            self.input_dim_,
                            self.hidden_dims_,
                            self.output_dim_, 1, 0)
        self.div_models_.append(mdl)

    def forward_train(self,
                      data: torch.Tensor,
                      data_shuffled: torch.Tensor) -> List[torch.Tensor]:
        # caluculation for P
        tmp_const_adjust_P = torch.Tensor([1]).type_as(
                          self.div_models_[0].layers_[0].weight)
        for _i_mdl in range(len(self.div_models_)-1):
            mdl = self.div_models_[_i_mdl]
            prd = mdl(data)
            prd_numerator_P = prd
            numerator_P = torch.exp(-prd_numerator_P)
            tmp_denominator_P = torch.mean(
                numerator_P)
            tmp_P = numerator_P/tmp_denominator_P
            tmp_const_adjust_P = tmp_const_adjust_P*tmp_P
        mean_tmp_const_adjust_P = torch.mean(
            tmp_const_adjust_P)
        const_previous_prob_adjusting_arr_P = (
            tmp_const_adjust_P
            / mean_tmp_const_adjust_P).detach()
        mdl = self.div_models_[len(self.div_models_)-1]
        preds_P = mdl(data)
        adjusting_prob_arr_P = (
          torch.exp(-p*preds_P)*const_previous_prob_adjusting_arr_P)

        preds_Q = mdl(data_shuffled)
        adjusting_prob_arr_Q = torch.exp(q*preds_Q)

        return adjusting_prob_arr_P, adjusting_prob_arr_Q

    def forward_estimate(self,
                         data: torch.Tensor) -> List[torch.Tensor]:
        # caluculation for P
        tmp_const_adjusting_arr_P = torch.Tensor([1]).type_as(
                                self.div_models_[0].layers_[0].weight)
        for _i_mdl in range(len(self.div_models_)):
            mdl = self.div_models_[_i_mdl]
            prd = mdl(data)
            prd_numerator_P = prd
            numerator_P = torch.exp(-prd_numerator_P)
            tmp_denominator_P = torch.mean(
                numerator_P)
            tmp_P = numerator_P/tmp_denominator_P
            tmp_const_adjusting_arr_P = tmp_const_adjusting_arr_P*tmp_P
        mean_tmp_const_adjusting_arr_P = torch.mean(
            tmp_const_adjusting_arr_P)
        const_adjusting_arr_P = (
            tmp_const_adjusting_arr_P
            / mean_tmp_const_adjusting_arr_P).detach()
        return const_adjusting_arr_P

    def forward(self,
                data: torch.Tensor,
                data_shuffled: torch.Tensor,
                pred_mode: str) -> List[torch.Tensor]:
        if pred_mode == 'train':
            return self.forward_train(data, data_shuffled)
        elif pred_mode == 'estimate_only':
            return self.forward_estimate(data)


class LightningCausalNets_DivEstimate(pl.LightningModule):
    def __init__(self,
                 params_div: dict,
                 learning_rate: float,
                 div_models_filePath: str
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate_ = learning_rate
        div_models = DivNetList(params_div)
        if os.path.exists(div_models_filePath):
            div_models_parameters = torch.load(div_models_filePath)
            keys_parameters = list(div_models_parameters.keys())
            modleid_list = [
                    int(re.findall(r'div_models_\.([0-9]+)\.*', _key)[0])
                    for _key in keys_parameters]
            n_models = max(modleid_list) + 1
            for _i_mdl in range(n_models):
                div_models.add_new_div_model()
            div_models.load_state_dict(div_models_parameters)

        div_models.add_new_div_model()
        self.div_models_ = div_models

    def forward(self, data, data_shuffled):
        return self.div_models_(data,
                                data_shuffled,
                                pred_mode='train')

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, data_shuffled = batch
        adjust_prob_P, adjust_prob_Q = self(data, data_shuffled)
        # numerator_Q = torch.exp(preds_Q)
        term_P = torch.mean(adjust_prob_P)
        term_Q = torch.mean(adjust_prob_Q)
        loss = term_P/p + term_Q/q
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        with torch.inference_mode():
            data, data_shuffled = batch
            adjust_prob_P, adjust_prob_Q = self(data, data_shuffled)
            term_P = torch.mean(adjust_prob_P)
            term_Q = torch.mean(adjust_prob_Q)
            loss = term_P/p + term_Q/q
            alpha_div = 1/(p*q) - loss
            self.log(f'validate_div', alpha_div,
                     on_step=True,
                     on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        with torch.inference_mode():
            data, data_shuffled = batch
            adjust_prob_P, adjust_prob_Q = self(data, data_shuffled)
            term_P = torch.mean(adjust_prob_P)
            term_Q = torch.mean(adjust_prob_Q)
            loss = term_P/p + term_Q/q
            alpha_div = 1/(p*q) - loss
            self.log(f'test_div', alpha_div,
                     on_step=True,
                     on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.div_models_.parameters(),
                          lr=self.learning_rate_)


def objective_gbt(
      X_train: Any,
      y_train: Any,
      X_test: Any,
      y_test: Any,
      sample_weight_train=None
      ) -> float:
    def _objective_gbt(trial: optuna.trial.Trial):
        N_data = X_train.shape[0]
        max_leaf_nodes = trial.suggest_int(
            'max_leaf_nodes', 2, int(N_data/2), log=True)
        learning_rate = trial.suggest_float('learning_rate', 0.0, 0.5)
        reg = GradientBoostingRegressor(
                random_state=0,
                max_leaf_nodes=max_leaf_nodes,
                subsample=RATE_SUBSUMPLE_TUNE_GDB,
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


def create_G(X_list, A_train, max_constract_dim):
    A_centered = A_train - np.mean(A_train)
    X_centered_list = []
    for _i_X in range(len(X_list)):
        Xi = X_list[_i_X]
        Xi_centered = Xi - np.mean(Xi)
        X_centered_list.append(Xi_centered)
    g_list = []
    for _i_X in range(len(X_list)):
        Xi_centered = X_centered_list[_i_X]
        for _p_x in range(max_constract_dim+1):
            if _p_x == 0:
                if _i_X == 0:
                    pXi_centered = np.ones(Xi_centered.shape)
                else:
                    continue
            else:
                pXi_centered = np.power(Xi_centered, _p_x)
            for _q_A in range(max_constract_dim+1):
                if _p_x == 0 and _q_A == 0:
                    continue
                else:
                    if _q_A == 0:
                        qA_centered = np.ones(A_centered.shape)
                    else:
                        qA_centered = np.power(A_centered, _q_A)
                pXiqA = pXi_centered*qA_centered
                g_list.append(pXiqA)
    G = np.concatenate(g_list, axis=1)
    return G


def f(lambda_vec, G):
    tmp = np.exp(- np.matmul(G, lambda_vec))
    return tmp.reshape(len(tmp), 1)


def objective_max_scalar(lambda_vec, G):
    res = np.sum(f(lambda_vec, G))
    return res


def jacobian_objective_max_scalar(lambda_vec, G):
    tmp = f(lambda_vec, G)
    res = - np.sum(G*tmp, axis=0)
    return res


"""
Run Simulations
"""
ohenc_Z5 = OneHotEncoder(sparse=False, drop='first')
all_results_mse_gbt_baseline = dict()
all_results_mse_gbt_entropy_balancing_dict = dict()
all_results_mse_gbt_ngd = dict()
all_results_mse_ln_baseline = dict()
all_results_mse_ln_entropy_balancing_dict = dict()
all_results_mse_ln_ngd = dict()
for _i_sim in range(N_SIMULATION):
    now = datetime.datetime.now()
    current_datetime = now.strftime('%Y%m%d_%H%M_%S')
    print(f'Start SIM={_i_sim:04}... --- current time: {current_datetime}')

    log_dir_name = os.path.join(
        out_parent_dir, 'all_logs', f'{_i_sim:03}_{current_datetime}')
    out_model_dir = os.path.join(log_dir_name, '_div_models')
    out_div_mdl_name = 'best_model.mi'
    os.makedirs(out_model_dir, exist_ok=True)

    # generating data
    denominator_Y = 50
    myu_vec = [-0.5, 1, 0, 1]
    sigma_mat = np.repeat(0.0, 4*4).reshape(4, 4)
    np.fill_diagonal(sigma_mat, 1)
    X_mat = np.random.multivariate_normal(myu_vec, sigma_mat, N_DATA_ALL)
    X1 = X_mat[:, 0]
    X2 = X_mat[:, 1]
    X3 = X_mat[:, 2]
    X4 = X_mat[:, 3]
    X5 = np.random.choice(a=3, size=N_DATA_ALL, p=(0.7, 0.15, 0.15))
    X5_mat = X5.reshape(N_DATA_ALL, 1)
    ohenc_Z5.fit(X5_mat)
    myuA = (5*np.abs(X1) + 6*np.abs(X2) + np.abs(X4)
            + 1*np.abs(X5 == 1) + 5*np.abs(X5 == 2))
    A = np.random.noncentral_chisquare(3, myuA)
    C = (1 + 3.5**2) + (1 + 24**2)
    Y = (((-0.15*A**2 + A*(X1**2+X2**2) - 15
          + (X1 + 3)**2 + 2*(X2 - 25)**2 + X3)
          - C)/denominator_Y + np.random.normal(0, 1, N_DATA_ALL))
    Z1 = np.exp(X1/2)
    Z2 = X2/(1 + np.exp(X1)) + 10
    Z3 = (X1*X3)/25 + 0.6
    Z4 = (X4 - 1)**2
    Z5 = ohenc_Z5.transform(X5_mat)

    Z1 = np.reshape(Z1.astype(np.float32),
                    (N_DATA_ALL, 1))
    Z2 = np.reshape(Z2.astype(np.float32),
                    (N_DATA_ALL, 1))
    Z3 = np.reshape(Z3.astype(np.float32),
                    (N_DATA_ALL, 1))
    Z4 = np.reshape(Z4.astype(np.float32),
                    (N_DATA_ALL, 1))
    Z5 = np.reshape(Z5.astype(np.float32),
                    (N_DATA_ALL, Z5.shape[1]))
    A = np.reshape(A.astype(np.float32),
                   (N_DATA_ALL, 1))
    Y = np.reshape(Y.astype(np.float32),
                   (N_DATA_ALL, 1))
    original_all_data_numpy_list_unit = [A, Z1, Z2, Z3, Z4, Z5, Y]
    n_unit_explanatories = len(original_all_data_numpy_list_unit) - 1

    unit_train_all_data_numpy_list = []
    unit_test_all_data_numpy_list = []
    for _idx in range(len(original_all_data_numpy_list_unit)):
        i_d = original_all_data_numpy_list_unit[_idx]
        i_train, i_test = train_test_split(
          i_d,
          test_size=N_DATA_TEST,
          train_size=N_DATA_TRAIN,
          random_state=0)
        unit_train_all_data_numpy_list.append(i_train)
        unit_test_all_data_numpy_list.append(i_test)

    Z_1_to_5_train_list = unit_train_all_data_numpy_list[1:6]
    Z_1_to_5_test_list = unit_test_all_data_numpy_list[1:6]
    Z_1_to_5_train = np.concatenate(Z_1_to_5_train_list, axis=1)
    Z_1_to_5_test = np.concatenate(Z_1_to_5_test_list, axis=1)

    train_all_data_numpy_list = ([unit_train_all_data_numpy_list[0]]
                                 + [Z_1_to_5_train]
                                 + unit_train_all_data_numpy_list[6:])

    test_all_data_numpy_list = ([unit_test_all_data_numpy_list[0]]
                                + [Z_1_to_5_test]
                                + unit_test_all_data_numpy_list[6:])

    train_data_tensor_list = []
    for _idx in range(len(train_all_data_numpy_list)):
        i_train = train_all_data_numpy_list[_idx]
        i_train_torch = torch.from_numpy(i_train)
        i_train_torch.requires_grad_(True)
        train_data_tensor_list.append(i_train_torch)

    n_explanatories = len(train_data_tensor_list) - 1
    response_tsr = train_data_tensor_list[
        len(train_data_tensor_list) - 1].detach()

    dim_n_explanatories = 0
    all_explanatories_train_tensor_list = []
    for _i_data in range(n_explanatories):
        i_data = train_data_tensor_list[_i_data]
        dim_n_explanatories += i_data.size()[1]
        all_explanatories_train_tensor_list.append(
          i_data.detach())
    all_explanatories_tsr = torch.cat(
        all_explanatories_train_tensor_list, dim=1)

    A_train = train_all_data_numpy_list[0]
    Z_train_list = train_all_data_numpy_list[
        1:(len(train_all_data_numpy_list) - 1)]
    y_train_all = train_all_data_numpy_list[
              len(train_all_data_numpy_list) - 1]
    Z_train_mat = np.concatenate(Z_train_list, axis=1)
    expls_train_all_mat = np.concatenate([A_train] + Z_train_list, axis=1)
    Z_uniqu_train_list = unit_train_all_data_numpy_list[
        1:(len(unit_train_all_data_numpy_list) - 1)]

    A_test = test_all_data_numpy_list[0]
    Z_test_list = test_all_data_numpy_list[
        1:(len(test_all_data_numpy_list) - 1)]

    Z_test_mat = np.concatenate(Z_test_list, axis=1)
    expls_test_mat = np.concatenate([A_test] + Z_test_list, axis=1)

    dropout_div = 0.0
    n_layers_div = N_LAYERS_DIV
    learning_rate_div = LEARNING_RATE
    hidden_dim_div = HIDDEN_DIM_DIV
    input_dim_div = dim_n_explanatories
    output_dim_div = 1
    params_div = {
        'dropout': dropout_div,
        'n_layers': n_layers_div,
        'input_dim': input_dim_div,
        'hidden_dim': hidden_dim_div,
        'output_dim': output_dim_div}

    # dummy
    current_best_div_estimated = np.Inf
    # dummy
    current_best_div_model_id = 99
    div_estimated_dict = {}
    div_molde_filepath_dict = {}
    for _i_div in range(0, N_DIV_MODELS):
        out_div_mdl_filename = f'div_model_{_i_div:02}.mi'
        if not np.isinf(current_best_div_estimated):
            now = datetime.datetime.now()
            logname_suffix = now.strftime('%Y-%m-%d_%H_%M_%S')
            logname = f'NBW_modeling({_i_div})_{logname_suffix}'
            print('----------------------------------------------------------------------')
            print(f' *** Start NBW modeling of {logname} ***')
            print('----------------------------------------------------------------------')
            print(f'paramas of NBW modeling: {params_div}')
            print('----------------------------------------------------------------------')

            all_explanatories_shuffled_list = []
            for _i_data in range(n_explanatories):
                i_data = train_data_tensor_list[_i_data]
                r = torch.randperm(N_DATA_TRAIN)
                i_data_suffuled = i_data[r, :]
                all_explanatories_shuffled_list.append(
                  i_data_suffuled.detach())
            explanatories_shuffled_tsr = torch.cat(
              all_explanatories_shuffled_list, dim=1)

            dataset_all_expls_to_fit = torch.utils.data.TensorDataset(
                all_explanatories_tsr, explanatories_shuffled_tsr)

            train_dataloaer_div = DataLoader(
                dataset_all_expls_to_fit,
                batch_size=BATCHSIZE
            )
            vaildate_dataloaer_div = DataLoader(
                dataset_all_expls_to_fit,
                batch_size=BATCHSIZE
            )
            test_dataloaer_div = DataLoader(
                dataset_all_expls_to_fit,
                batch_size=BATCHSIZE
            )

            current_best_div_model_filename = (
              f'div_model_{current_best_div_model_id:02}.mi')
            current_best_div_model_filepath = os.path.join(
                out_model_dir, current_best_div_model_filename)
            model_div = LightningCausalNets_DivEstimate(
                              params_div,
                              learning_rate_div,
                              current_best_div_model_filepath
                              )

            logger = TensorBoardLogger(
                log_dir_name,
                logname,
                default_hp_metric=False
            )
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=2,
                strategy='dp',
                logger=logger,
                max_epochs=EPOCHS_DIV,
                enable_progress_bar=False
            )

            trainer.fit(model_div,
                        train_dataloaer_div,
                        vaildate_dataloaer_div)

            trainer.test(model_div, test_dataloaer_div)

            trainer.logger.log_hyperparams(params_div)

            test_val_name = 'test_div'
            test_val = trainer.callback_metrics[test_val_name].item()
            trainer.logger.log_hyperparams(
                params_div,
                {test_val_name: test_val})

            out_div_mdl_filepath = os.path.join(
                out_model_dir, out_div_mdl_filename)
            torch.save(model_div.div_models_.state_dict(),
                       out_div_mdl_filepath)
            div_molde_filepath_dict[_i_div] = out_div_mdl_filepath

            del model_div
            del explanatories_shuffled_tsr

        """
        Estimating alpha-information
        (mesuring the performance of the previous MBW model)
        """
        now = datetime.datetime.now()
        logname_suffix = now.strftime('%Y-%m-%d_%H_%M_%S')
        logname = f'alpha-info_estimating({_i_div})_{logname_suffix}'
        print('----------------------------------------------------------------------')
        print(f' *** Start alpha-info modeling of {logname} ***')
        print('----------------------------------------------------------------------')
        print(f'paramas of alpha-info modeling {params_div}')
        print('----------------------------------------------------------------------')

        all_explanatories_shuffled_list = []
        for _i_data in range(n_explanatories):
            i_data = train_data_tensor_list[_i_data]
            r = torch.randperm(N_DATA_TRAIN)
            i_data_suffuled = i_data[r, :]
            all_explanatories_shuffled_list.append(
              i_data_suffuled.detach())
        explanatories_shuffled_tsr = torch.cat(
          all_explanatories_shuffled_list, dim=1)

        dataset_all_expls_to_fit = torch.utils.data.TensorDataset(
            all_explanatories_tsr, explanatories_shuffled_tsr)

        train_dataloaer_div = DataLoader(
            dataset_all_expls_to_fit,
            batch_size=BATCHSIZE
        )
        vaildate_dataloaer_div = DataLoader(
            dataset_all_expls_to_fit,
            batch_size=BATCHSIZE
        )
        test_dataloaer_div = DataLoader(
            dataset_all_expls_to_fit,
            batch_size=BATCHSIZE
        )

        previous_div_model_filepath = os.path.join(
            out_model_dir, out_div_mdl_filename)
        model_div = LightningCausalNets_DivEstimate(
                          params_div,
                          learning_rate_div,
                          previous_div_model_filepath)

        logger = TensorBoardLogger(
            log_dir_name,
            logname,
            default_hp_metric=False
        )
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=2,
            strategy='dp',
            logger=logger,
            max_epochs=EPOCHS_DIV,
            enable_progress_bar=False
            )

        trainer.fit(model_div,
                    train_dataloaer_div,
                    vaildate_dataloaer_div)

        trainer.test(model_div, test_dataloaer_div)

        trainer.logger.log_hyperparams(params_div)
        test_val_name = 'test_div'
        test_val = trainer.callback_metrics[test_val_name].item()
        trainer.logger.log_hyperparams(
            params_div,
            {test_val_name: test_val})

        adjust_prob_P, adjust_prob_Q = model_div(
            all_explanatories_tsr, explanatories_shuffled_tsr)

        loss = (
          torch.mean(adjust_prob_P)/p + torch.mean(adjust_prob_Q)/q).item()
        div_estimated_raw = 1/(p*q) - loss
        div_estimated_dict[_i_div] = div_estimated_raw
        div_estimated = abs(div_estimated_raw)
        if np.isinf(current_best_div_estimated):
            if div_estimated >= DIVN_INITNAIL_DIV:
                current_best_div_model_id = _i_div
                current_best_div_estimated = div_estimated
        else:
            if div_estimated < current_best_div_estimated*0.99:
                current_best_div_model_id = _i_div
                current_best_div_estimated = div_estimated

        print('----------------------------------------------------------------------')
        print(f'NBW model: ID = {_i_div}, ',
              f'alpha-information = {div_estimated}')
        print(f'Best NBW model: ID = {current_best_div_model_id}, ',
              f'alpha-information = {current_best_div_estimated}')
        print('-------------------------------------------------------------------------')

        del model_div
        del explanatories_shuffled_tsr

    print(
      '-----------------------------------------------------------------------')
    print(
      '-----------------------------------------------------------------------')

    print(f'Best NBW model: ID = {current_best_div_model_id}, ',
          f'alpha-information = {current_best_div_estimated}')
    print(
      '-----------------------------------------------------------------------')
    print(
      '-----------------------------------------------------------------------')

    expls_train, expls_test, y_train, y_test = train_test_split(
        expls_train_all_mat, y_train_all, test_size=0.25, random_state=0)

    opt_div_model_path = div_molde_filepath_dict[current_best_div_model_id]
    div_models = DivNetList(params_div)
    div_models_parameters = torch.load(opt_div_model_path)
    keys_parameters = list(div_models_parameters.keys())
    modleid_list = [
            int(re.findall(r'div_models_\.([0-9]+)\.*', _key)[0])
            for _key in keys_parameters]
    n_models = max(modleid_list) + 1
    for _i_mdl in range(n_models):
        div_models.add_new_div_model()
    div_models.load_state_dict(div_models_parameters)

    expls_train_tsr = torch.from_numpy(expls_train)
    const_prob_adjusting_train_tsr = div_models(
        expls_train_tsr, None, 'estimate_only')

    expls_test_tsr = torch.from_numpy(expls_test)
    const_prob_adjusting_test_tsr = div_models(
        expls_test_tsr, None, 'estimate_only')

    const_prob_adjusting_train_mat = const_prob_adjusting_train_tsr.numpy()
    const_prob_adjusting_train_arr = const_prob_adjusting_train_mat.flatten()

    """
    No weights hyperparameter searches for GDB
    """
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(
        objective_gbt(
          expls_train,
          y_train,
          expls_test,
          y_test,
        ),
        n_trials=N_TRY_OPT,
        timeout=OPTUNA_TIMEOUT_SECONDS
    )
    best_trial_baseline = study_baseline.best_trial
    print(best_trial_baseline)

    """
    NBW hyperparameter searches for GDB
    """
    study_ngd = optuna.create_study(direction='minimize')
    study_ngd.optimize(
        objective_gbt(
          expls_train,
          y_train,
          expls_test,
          y_test,
          const_prob_adjusting_train_arr
          ),
        n_trials=N_TRY_OPT,
        timeout=OPTUNA_TIMEOUT_SECONDS
    )
    best_trial_ngd = study_ngd.best_trial
    print(best_trial_ngd)

    """
    Entropy Balancing hyperparameter searches for GDB
    """
    bounds = scipy.optimize.Bounds(-300, 300)
    reg_gbt_entropy_balancing_dic = dict()
    weight_vec_all_dict = dict()
    for _p in range(1, MAX_DEGREE_OF_POLY):
        G = create_G(Z_uniqu_train_list, A_train, _p)
        lambda0_vec = np.repeat(0.000001, G.shape[1])
        res_optimize = scipy.optimize.minimize(
          objective_max_scalar,
          x0=lambda0_vec,
          args=(G),
          method='L-BFGS-B',
          bounds=bounds,
          jac=jacobian_objective_max_scalar)
        print(f'optimize status: {res_optimize.success}')

        temp_weihgt = f(res_optimize.x, G)
        weight_mat = temp_weihgt/np.mean(temp_weihgt)
        weight_vec_all = weight_mat.flatten()
        weight_vec_all_dict[_p] = weight_vec_all

        expls_train, expls_test, y_train, y_test, weight_vec_train, _ = (
          train_test_split(
            expls_train_all_mat, y_train_all, weight_vec_all,
            test_size=0.25, random_state=0))

        study_entropy_balancing = optuna.create_study(direction='minimize')

        const_weight_vec_train = weight_vec_train/np.mean(weight_vec_train)
        study_entropy_balancing.optimize(
            objective_gbt(
              expls_train,
              y_train,
              expls_test,
              y_test,
              const_weight_vec_train
              ),
            n_trials=N_TRY_OPT,
            timeout=OPTUNA_TIMEOUT_SECONDS)

        best_trial_entropy_balancing = study_entropy_balancing.best_trial
        print(best_trial_entropy_balancing)

        learning_rate_entropy_balancing = best_trial_entropy_balancing.params[
            'learning_rate']
        max_leaf_nodes_entropy_balancing = best_trial_entropy_balancing.params[
            'max_leaf_nodes']
        reg_gbt_entropy_balancing = GradientBoostingRegressor(
                random_state=0,
                max_leaf_nodes=max_leaf_nodes_entropy_balancing,
                learning_rate=learning_rate_entropy_balancing)
        reg_gbt_entropy_balancing.fit(
            expls_train_all_mat, y_train_all, sample_weight=weight_vec_all)
        reg_gbt_entropy_balancing_dic[_p] = reg_gbt_entropy_balancing

    const_prob_adjusting_fit_tsr = div_models(
        all_explanatories_tsr, None, 'estimate_only')
    const_prob_adjusting_fit_mat = const_prob_adjusting_fit_tsr.numpy()
    const_prob_adjusting_fit_arr = const_prob_adjusting_fit_mat.flatten()

    learning_rate_baseline = best_trial_baseline.params['learning_rate']
    max_leaf_nodes = best_trial_baseline.params['max_leaf_nodes']
    reg_gbt_baseline = GradientBoostingRegressor(
            random_state=0,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate_baseline)
    reg_gbt_baseline.fit(expls_train_all_mat, y_train_all)

    learning_rate_ngd = best_trial_ngd.params['learning_rate']
    max_leaf_nodes_ngd = best_trial_ngd.params['max_leaf_nodes']
    reg_gbt_ngd = GradientBoostingRegressor(
            random_state=0,
            max_leaf_nodes=max_leaf_nodes_ngd,
            learning_rate=learning_rate_ngd)
    reg_gbt_ngd.fit(
        expls_train_all_mat, y_train_all,
        sample_weight=const_prob_adjusting_fit_arr)

    # Estimating causal effect
    r_index_A = np.random.choice(len(A_test), size=len(A_test), replace=False)
    A_intervention = A_test[r_index_A, :]

    r1_index_Z = np.random.choice(
        len(Z_test_mat), size=len(Z_test_mat),
        replace=False)
    Z_test = Z_test_mat[r1_index_Z, :]
    Z1_intervention = Z_test[:, [0]]
    Z2_intervention = Z_test[:, [1]]
    Z3_intervention = Z_test[:, [2]]
    r2_index_Z = np.random.choice(
        len(Z_test_mat), size=len(Z_test_mat),
        replace=False)
    Z4_test = Z_test[:, [3]]
    Z4_intervention = Z4_test[r2_index_Z, :]

    r3_index_Z = np.random.choice(
        len(Z_test_mat), size=len(Z_test_mat),
        replace=False)
    Z5_test = Z_test[:, 4:]
    Z5_intervention = Z5_test[r3_index_Z, :]

    expls_intervention_mat = np.concatenate(
      [A_intervention,
       Z1_intervention,
       Z2_intervention,
       Z3_intervention,
       Z4_intervention,
       Z5_intervention], axis=1)

    X1_intervention = 2*np.log(Z1_intervention.flatten())
    X2_intervention = (Z2_intervention.flatten() - 10)*(
      1+Z1_intervention.flatten()**2)
    X3_intervention = 25*(Z3_intervention.flatten() - 0.6)/X1_intervention

    Y_true_intervention = (-0.15*A_intervention.flatten()**2
                           + A_intervention.flatten()*(
                                X1_intervention**2
                                + X2_intervention**2)
                           - 15
                           + (X1_intervention + 3)**2
                           + 2*(X2_intervention - 25)**2
                           + X3_intervention
                           - C)/denominator_Y

    Y_estimated_gbt_baseline = reg_gbt_baseline.predict(
        expls_intervention_mat)
    mse_gbt_baseline = mean_squared_error(
        Y_estimated_gbt_baseline, Y_true_intervention)

    mse_gbt_entropy_balancing_dict = dict()
    rate_gdt_entropy_balancing_dict = dict()
    for _p in reg_gbt_entropy_balancing_dic.keys():
        reg_gbt_entropy_balancing = reg_gbt_entropy_balancing_dic[_p]
        Y_estimated_gbt_entropy_balancing = (
          reg_gbt_entropy_balancing.predict(expls_intervention_mat))
        mse_gbt_entropy_balancing = mean_squared_error(
            Y_estimated_gbt_entropy_balancing, Y_true_intervention)
        mse_gbt_entropy_balancing_dict[_p] = mse_gbt_entropy_balancing
        rate_gdt_entropy_balancing_dict[_p] = (
          mse_gbt_entropy_balancing/mse_gbt_baseline)

    Y_estimated_gbt_ngd = reg_gbt_ngd.predict(
        expls_intervention_mat)
    mse_gbt_ngd = mean_squared_error(
        Y_estimated_gbt_ngd, Y_true_intervention)

    print(
      '-----------------------------------------------------------------------')
    print(f'GBT MSE(No weights)={mse_gbt_baseline}')
    print(f'GBT MSE(Entopy Balancing)={mse_gbt_entropy_balancing_dict}')
    print(f'GBT MSE(NBW)={mse_gbt_ngd}')
    print(
      '-----------------------------------------------------------------------')
    print(
      f'GBT MSE Ratio (Entropy Balancing(4) / No weights) = ',
      f'{rate_gdt_entropy_balancing_dict}')
    print(
      f'GBT MSE Ratio (NBW / weights) = ',
      f'{mse_gbt_ngd/mse_gbt_baseline}')
    print(
      '-----------------------------------------------------------------------')

    # saving results
    all_results_mse_gbt_baseline[_i_sim] = mse_gbt_baseline
    all_results_mse_gbt_entropy_balancing_dict[_i_sim] = (
        mse_gbt_entropy_balancing_dict)
    all_results_mse_gbt_ngd[_i_sim] = mse_gbt_ngd

    reg_ln_baseline = LinearRegression().fit(
      expls_train_all_mat, y_train_all)
    Y_estimated_ln_baseline = reg_ln_baseline.predict(
            expls_intervention_mat)
    mse_ln_baseline = mean_squared_error(
        Y_estimated_ln_baseline.flatten(), Y_true_intervention)

    mse_ln_entropy_balancing_dict = dict()
    rate_ln_entropy_balancing_dict = dict()
    for _p in reg_gbt_entropy_balancing_dic.keys():
        weight_vec_all = weight_vec_all_dict[_p]
        reg_ln_entropy_balancing = LinearRegression().fit(
          expls_train_all_mat, y_train_all, sample_weight=weight_vec_all)
        Y_estimated_ln_entropy_balancing = reg_ln_entropy_balancing.predict(
            expls_intervention_mat)
        mse_ln_entropy_balancing = mean_squared_error(
          Y_estimated_ln_entropy_balancing, Y_true_intervention)
        mse_ln_entropy_balancing_dict[_p] = mse_ln_entropy_balancing
        rate_ln_entropy_balancing_dict[_p] = (
            mse_ln_entropy_balancing/mse_ln_baseline)

    reg_ln_ngd = LinearRegression().fit(
        expls_train_all_mat, y_train_all,
        sample_weight=const_prob_adjusting_fit_arr)
    Y_estimated_ln_ngd = reg_ln_ngd.predict(expls_intervention_mat)
    mse_ln_ngd = mean_squared_error(
        Y_estimated_ln_ngd, Y_true_intervention)

    print(
      '-----------------------------------------------------------------------')
    print(f'Linear MSE (No weights)={mse_ln_baseline}')
    print(f'Linear MSE (Entropy Balancing)={mse_ln_entropy_balancing_dict}')
    print(f'Linear MSE (NBW)={mse_ln_ngd}')
    print(
       '-----------------------------------------------------------------------')
    print(f'Linear MSE Ratio (Entropy Balancing(4) / No weights) = ',
          f'{rate_ln_entropy_balancing_dict}')
    print(f'Linear MSE Ratio (NBW / No weights) = ',
          f'{mse_ln_ngd/mse_ln_baseline}')
    print(
      '-----------------------------------------------------------------------')

    # saving results
    all_results_mse_ln_baseline[_i_sim] = mse_ln_baseline
    all_results_mse_ln_entropy_balancing_dict[_i_sim] = (
        mse_ln_entropy_balancing_dict)
    all_results_mse_ln_ngd[_i_sim] = mse_ln_ngd
    # ---

    del X1, X2, X3, X4, X5, X5_mat
    del Y, A,
    del Z1, Z2, Z3, Z4, Z5
    del Z_train_mat, expls_train_all_mat
    del Z_test_mat, expls_test_mat
    del train_data_tensor_list
    del response_tsr, all_explanatories_tsr

# output results (binary file)
#  --- all_results_mse_gbt_baseline
out_fine_name_base = 'all_results_mse_gbt_baseline'
result_to_output = all_results_mse_gbt_baseline
with open(
      os.path.join(out_result_dir, out_fine_name_base + '.pickle'), 'wb') as wf:
    pickle.dump(result_to_output, wf)

#  --- all_results_mse_gbt_entropy_balancing_dict
out_fine_name_base = 'all_results_mse_gbt_entropy_balancing_dict'
result_to_output = all_results_mse_gbt_entropy_balancing_dict
with open(
      os.path.join(out_result_dir, out_fine_name_base + '.pickle'), 'wb') as wf:
    pickle.dump(result_to_output, wf)

#  --- all_results_mse_gbt_ngd
out_fine_name_base = 'all_results_mse_gbt_ngd'
result_to_output = all_results_mse_gbt_ngd
with open(
      os.path.join(out_result_dir, out_fine_name_base + '.pickle'), 'wb') as wf:
    pickle.dump(result_to_output, wf)

#  --- all_results_mse_ln_baseline
out_fine_name_base = 'all_results_mse_ln_baseline'
result_to_output = all_results_mse_ln_baseline
with open(
      os.path.join(out_result_dir, out_fine_name_base + '.pickle'), 'wb') as wf:
    pickle.dump(result_to_output, wf)

#  --- all_results_mse_ln_entropy_balancing_dict
out_fine_name_base = 'all_results_mse_ln_entropy_balancing_dict'
result_to_output = all_results_mse_ln_entropy_balancing_dict
with open(
      os.path.join(out_result_dir, out_fine_name_base + '.pickle'), 'wb') as wf:
    pickle.dump(result_to_output, wf)

#  --- all_results_mse_ln_ngd
out_fine_name_base = 'all_results_mse_ln_ngd'
result_to_output = all_results_mse_ln_ngd
with open(
      os.path.join(out_result_dir, out_fine_name_base + '.pickle'), 'wb') as wf:
    pickle.dump(result_to_output, wf)


# concat all results
res_all_df_list = []
for _i_row in range(len(all_results_mse_ln_baseline)):
    # Linear
    row_linear_unweighted = pd.DataFrame(
      {'Linear_unweighted': np.array([all_results_mse_ln_baseline[_i_row]])})
    row_linear_EP = pd.DataFrame({
        'Linear_EP(1)': np.array(
          [all_results_mse_ln_entropy_balancing_dict[_i_row][1]]),
        'Linear_EP(2)': np.array(
          [all_results_mse_ln_entropy_balancing_dict[_i_row][2]]),
        'Linear_EP(3)': np.array(
          [all_results_mse_ln_entropy_balancing_dict[_i_row][3]]),
        'Linear_EP(4)': np.array([
          all_results_mse_ln_entropy_balancing_dict[_i_row][4]])})
    row_linear_NBW = pd.DataFrame(
      {'Linear_NBW': np.array([all_results_mse_ln_ngd[_i_row]])})

    # GDT
    row_gdt_unweighted = pd.DataFrame(
      {'GBT_unweighted': np.array([all_results_mse_gbt_baseline[_i_row]])})
    row_gdt_EP = pd.DataFrame({
        'GBT_EP(1)': np.array(
            [all_results_mse_gbt_entropy_balancing_dict[_i_row][1]]),
        'GBT_EP(2)': np.array(
            [all_results_mse_gbt_entropy_balancing_dict[_i_row][2]]),
        'GBT_EP(3)': np.array(
            [all_results_mse_gbt_entropy_balancing_dict[_i_row][3]]),
        'GBT_EP(4)': np.array(
            [all_results_mse_gbt_entropy_balancing_dict[_i_row][4]])})
    row_gdt_NBW = pd.DataFrame(
      {'GBT_NBW': np.array([all_results_mse_gbt_ngd[_i_row]])})
    row_all_res = pd.concat(
      [row_linear_unweighted, row_linear_EP, row_linear_NBW,
       row_gdt_unweighted, row_gdt_EP, row_gdt_NBW], axis=1)
    res_all_df_list.append(row_all_res)

res_all_df = pd.concat(res_all_df_list)
res_all_df.reset_index(drop=True, inplace=True)

tmp_res_summary_df = pd.DataFrame(
  res_all_df.apply(func=_calc_stas, axis=0))
tmp_res_summary_df.rename(index={0:'mean', 1:'std'}, inplace=True)
res_summary_df = tmp_res_summary_df.transpose()

# output results (csv file)
res_all_df.to_csv(
    os.path.join(
      out_result_dir, Log_dir_headname + '_all.csv'), index=True)
res_summary_df.to_csv(
    os.path.join(
      out_result_dir, Log_dir_headname + '_summary.csv'), index=True)
