import os
import pickle
import datetime

from typing import List, Any, Tuple
import numpy as np

import torch
from torch import nn, Tensor
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class NbwDense(nn.Module):
    """Primitive neural network class for calculating the balancing weights.

    A densely-connected neural network class for calculating the balancing
    weights. From the constructor of this class, an instance is created as
    follows:
      * The size of an input layer is “input_dim”
      * Each size of hidden layers is “hidden_dims”

    Attributes:
      layers_: densely-connected neural network created in the structure
      described above.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 ):
        """Initializes an instance of this class.

        Args:
          input_dim: size of an input layer.
          hidden_dims: each size of hidden layers.
        """
        super().__init__()
        layers: List[nn.Module] = []
        _h_input_dim: int = input_dim
        for _h_out_dim in hidden_dims:
            layers.append(nn.Linear(_h_input_dim, _h_out_dim))
            layers.append(nn.ReLU())
            _h_input_dim = _h_out_dim
        output_dim = 1
        layers.append(nn.Linear(_h_input_dim, output_dim))
        self.layers_: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layers_(data)


class NbwDenseModelList(nn.Module):
    """Neural network class for a list of Neural Balancing Weight (NBW)
       models, where each of NBW models is an instance of NbwDense class.

      An instance of this class can have more than one instance of NbwDense
      class, where each instance of NbwDense class is obtained from enhancing
      the balancing ability of the previous weights.

    Attributes:
      alpha: Alpha value for alpha divergence.
      input_dim_: Size of an input layer for each of instances of NbwDense
                  class.
      hidden_dim_: Size of hedden layers for each of instances of NbwDense
                   class.
      n_layers_: Number of layers for each of instances of NbwDense class.
                 nbw_models_: List (nn.ModuleList) of all instances of NbwDense
                 class.
      new_nbw_model_: The last instance of NbwDense added to the “nbw_models_”
                      attribute. This object is a NGB model obtained from
                      enhancing the balancing ability of the models of
                      nbw_models_[0:(len(nbw_models_)- 1)].
    """

    def __init__(self,
                 alpha: float,
                 params_for_nbw_models: dict):
        """Initializes an instance of this class.

        Args:
          alpha: Alpha value for alpha divergence.
          params_for_nbw_models: Dictionary which contains the folowing
          parameters for creating an instance of NbwDense class.
            * input_dim_: Size of an input layer for each of instancess
                          of NbwDense class.
            * hidden_dim_: Size of hedden layers for each of instances of
                           NbwDense class.
            * n_layers: Number of layers for each of instances of
                        NbwDense class.
        """
        super().__init__()
        self.alpha_ = alpha
        self.input_dim_ = params_for_nbw_models['input_dim']
        self.hidden_dim_ = params_for_nbw_models['hidden_dim']
        self.n_layers_ = params_for_nbw_models['n_layers']
        self.hidden_dims_ = [self.hidden_dim_] * (self.n_layers_ - 1)
        self.nbw_models_ = nn.ModuleList()
        self.new_nbw_model_ = None

    def add_model(self):
        """Create a new instance of NbwDense class, and add it to the last
           of “nbw_models_” """
        mdl = NbwDense(self.input_dim_,
                       self.hidden_dims_)
        self.new_nbw_model_ = mdl
        self.nbw_models_.append(self.new_nbw_model_)

    def load_and_add_model(self, nbw_model_parameters: dict):
        """Create a new instance of NbwDense class whose neural network
           parameters (i.e. weights of the neural network) are given from
           “nbw_model_parameters”. Here, “nbw_model_parameters” is an object
           returned from state_dict() function of an instance of NbwDense
           class.
        """
        mdl = NbwDense(self.input_dim_,
                       self.hidden_dims_)
        mdl.load_state_dict(nbw_model_parameters)
        self.new_nbw_model_ = mdl
        self.nbw_models_.append(self.new_nbw_model_)

    def _forward_train(self,
                       data: torch.Tensor,
                       data_shuffled: torch.Tensor) -> List[torch.Tensor]:
        tmp_const_adjust_P = torch.Tensor([1]).type_as(
                          self.nbw_models_[0].layers_[0].weight)
        for _i_mdl in range(len(self.nbw_models_)-1):
            mdl = self.nbw_models_[_i_mdl]
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
        mdl = self.nbw_models_[len(self.nbw_models_)-1]
        preds_P = mdl(data)
        adjusting_prob_arr_P = (
          torch.exp(
            (self.alpha_ - 1)*preds_P)*const_previous_prob_adjusting_arr_P)

        preds_Q = mdl(data_shuffled)
        adjusting_prob_arr_Q = torch.exp((1 - self.alpha_)*preds_Q)

        return adjusting_prob_arr_P, adjusting_prob_arr_Q

    def _forward_estimate(self,
                          data: torch.Tensor) -> List[torch.Tensor]:
        tmp_const_adjusting_arr_P = torch.Tensor([1]).type_as(
                                self.nbw_models_[0].layers_[0].weight)
        for _i_mdl in range(len(self.nbw_models_)):
            mdl = self.nbw_models_[_i_mdl]
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
        """Forwad propagation function

        If pred_mode = "train", this function propagates foward two tensors for
        training a NBW model, which are used for calculating a loss we proposed
        in our paper for estimating alpha divergence.
        On the other hand, if pred_mode = "estimate_only", this propagates
        foward one tensor for caluclating the balancing weights. For the
        details, see the description of Returns below.

        Args:
          data: Training data.
          data_shuffled: Data obtained from shuffling each valiable of training
          data by the index.
          pred_mode:
              "train" or "estimate_only", which specifies a mode of return
              values. See the description below of Returns for the details.

        Returns:
          If pred_mode is "train", this function propagates foward two
          tensors,  e^{alpha*T(X)} under Q and e^{(alpha - 1)*T(X)} under P,
          where P is the distribution balanced by the previous NBW models and
          Q is the target distribution of balancing.

          If pred_mode is "estimate_only", this function propagates foward one
          tesor, e^{-T(X)} under P.
        """
        if pred_mode == 'train':
            return self._forward_train(data, data_shuffled)
        elif pred_mode == 'estimate_only':
            return self._forward_estimate(data)


class LightningNbwDense(pl.LightningModule):
    """Class for traing a Neural Balancing Weight (NBW) model.

      An instance of this class is used in train_and_enhance_NBW() or
      build_new_nbw_model() defined below, where the two funtion train a NBW
      model under the distribution balanced by the previous NBW models. (Here,
      a NBW model is an instance of NbwDense class.)

    Attributes:
      alpha_: Alpha value for alpha divergence.
      learning_rate_ :
          Learning rate for mini-batch stochastic gradient optimization in
          training a NBW model.
      nbw_models_:
          An instance of NbwDenseModelList class used for calculating balancing
          weights.
    """

    def __init__(self,
                 alpha: float,
                 params_nbw: dict,
                 learning_rate: float,
                 previous_nbw_model_filePath_list: List[str]
                 ):
        """Initializes an instance of this class.

        Args:
          alpha: Alpha value for alpha divergence.
          params_nbw:
              Dictionary which contains the folowing parameters for creating an
              instance of NbwDense class.
                * input_dim_: Size of an input layer for each of instances
                              of NbwDense class.
                * hidden_dim_: Size of hedden layers for each of instances of
                              NbwDense class.
                * n_layers: Number of layers for each of instances of
                            NbwDense class.
          learning_rate:
              Learning rate for mini-batch stochastic gradient optimization in
              training a NBW model.
          previous_nbw_model_filePath_list:
              List of filipathes of previous NBW models (binary files of
              instances of NbwDense class) built and saved for enchaning thier
              balancing ability. After loading the models in this list, they
              are used to balance the distribution of the training data. Then,
              a new NGB model is trained under the balanced distribution.
        """
        super().__init__()
        self.save_hyperparameters()
        self.alpha_ = alpha
        self.learning_rate_ = learning_rate
        nbw_modellist = NbwDenseModelList(self.alpha_, params_nbw)
        for _i_mdl_path in range(len(previous_nbw_model_filePath_list)):
            pre_nbw_mdl_fpath = previous_nbw_model_filePath_list[_i_mdl_path]
            if os.path.exists(pre_nbw_mdl_fpath):
                nbw_model_parameters = torch.load(pre_nbw_mdl_fpath)
                nbw_modellist.load_and_add_model(nbw_model_parameters)
        nbw_modellist.add_model()
        self.nbw_models_ = nbw_modellist

    def forward(self, data, data_shuffled):
        """Forwad propagation

        Args:
          data: Training data.
          data_shuffled:
              Data obtained from shuffling each valiable of training data by
              the index.

        Returns:
           This function propagates foward two tensors,  e^{alpha*T(X)} under
           Q and e^{(alpha - 1)*T(X)} under P, where P is the distribution
           balanced by the previous NBW models and Q is the target distribution
           of balancing.
        """
        return self.nbw_models_(data,
                                data_shuffled,
                                pred_mode='train')

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, data_shuffled = batch
        adjust_prob_P, adjust_prob_Q = self(data, data_shuffled)
        term_P = torch.mean(adjust_prob_P)
        term_Q = torch.mean(adjust_prob_Q)
        loss = term_P/self.alpha_ + term_Q/(1 - self.alpha_)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        with torch.inference_mode():
            data, data_shuffled = batch
            adjust_prob_P, adjust_prob_Q = self(data, data_shuffled)
            term_P = torch.mean(adjust_prob_P)
            term_Q = torch.mean(adjust_prob_Q)
            loss = term_P/self.alpha_ + term_Q/(1 - self.alpha_)
            alpha_div = 1/(self.alpha_*(1 - self.alpha_)) - loss
            self.log(f'validate_alpha_infomation', alpha_div,
                     on_step=True,
                     on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        with torch.inference_mode():
            data, data_shuffled = batch
            adjust_prob_P, adjust_prob_Q = self(data, data_shuffled)
            term_P = torch.mean(adjust_prob_P)
            term_Q = torch.mean(adjust_prob_Q)
            loss = term_P/self.alpha_ + term_Q/(1 - self.alpha_)
            alpha_div = 1/(self.alpha_*(1 - self.alpha_)) - loss
            self.log(f'test_alpha_infomation', alpha_div,
                     on_step=True,
                     on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.nbw_models_.parameters(),
                          lr=self.learning_rate_)


def load_nbw_models(
            alpha: float,
            params_nbw: dict,
            previous_nbw_model_filePath_list: List[str]
        ) -> NbwDenseModelList:
    """Load Neural Balancing Weight (NBW) model files (binary files of
       instances of NbwDense class) and return it as an instance of
       NbwDenseModelList.

    Args:
      alpha: Alpha value for alpha divergence.
      params_nbw: Dictionary which contains the folowing parameters for
      creating an instance of NbwDense class.
        * input_dim_: Size of an input layer for each of instances
                      of NbwDense class.
        * hidden_dim_: Size of hedden layers for each of instances of
                        NbwDense class.
        * n_layers: Number of layers for each of instances of
                    NbwDense class.
      previous_nbw_model_filePath_list:
          List of filipathes of the previous NBW models (binary files of
          instances of NbwDense class) built and saved for enchaning thier
          balancing.

    Returns:
        An instance of NbwDenseModelList created by loading NBW model files
        in "previous_nbw_model_filePath_list".
    """
    nbw_modellist = NbwDenseModelList(alpha, params_nbw)
    for _i_mdl_path in range(len(previous_nbw_model_filePath_list)):
        pre_nbw_mdl_fpath = previous_nbw_model_filePath_list[_i_mdl_path]
        if os.path.exists(pre_nbw_mdl_fpath):
            nbw_model_parameters = torch.load(pre_nbw_mdl_fpath)
            nbw_modellist.load_and_add_model(nbw_model_parameters)

    return nbw_modellist


def build_new_nbw_model(
            alpha: float,
            explanatories_to_be_balanced: List[Any],
            params_nbw: dict,
            learning_rate: float,
            batchsize: int,
            pytorch_trainer: pl.Trainer,
            previous_nbw_model_filePath_list: List[str]
        ) -> [NbwDense, float]:
    """Built a Neural Balancing Weight (NBW) model (an instance of
       NbwDense class) under the distribution balanced by the previous
       NBW models, and then return results of building the model.

    This function Creates and trains a Neural Balancing Weight (NBW)
    model, and then return results of training the model.

    In tarining the model, this function load the previous NBW models
     (binary files of instances of NbwDense class) saved at
    previous_nbw_model_filePath_list”. These models are used to balance the
    distribution of the learning data, “explanatories_to_be_balanced”.
    Then, a new NGB model is trained under the balanced distribution, and
    this function returns the model. In addition, this function also returns
    an estimate of the alpha-information of the the balanced distribution
    which is obtained from predictions of the model trained.

    Args:
      alpha: Alpha value for alpha divergence.
      explanatories_to_be_balanced:
          List of 2-dimensional numpy arrays. Each of numpy arrays in the
          list corresponds to an explanatory variable of training data,
          and has rows with the same size and columns with various sizes.
      params_nbw: Dictionary which contains the folowing parameters for
                  creating an instance of NbwDense class.
            * input_dim_: Size of an input layer for each of instances
                          of NbwDense class.
            * hidden_dim_: Size of hedden layers for each of instances of
                           NbwDense class.
            * n_layers: Number of layers for each of instances of NbwDense
                        class.
      learning_rate: Learning rate for mini-batch stochastic gradient
                     optimization in training a NBW model.
      batchsize: Batchsize for mini-batch stochastic gradient optimization
                 in training a NBW model.
      pytorch_trainer: Pytorch lightning trainer used for training a NBW
                       model.
      previous_nbw_model_filePath_list:
          List of filipathes of previous NBW models (binary files of
          instances of NbwDense class) .  The models in this list are
          used to balance the distribution of the learning data. A new NGB
          model is trained under the balanced distribution.


    Returns:
      new_nbw_model: A new NGB model (instances of NbwDense class) trained
                     under the distribution balanced by the weights from the
                     previous NBW models.
      alpha_infomation: An estimate of the alpha-information of the
      distribution balanced by the weights from the previous NBW models,
      which is obtained from predictions of the new model.
    """

    # Convert numpy arrays to pytoch tensor
    explanatories_tensor_list = []
    for _i_exp in range(len(explanatories_to_be_balanced)):
        tmp = explanatories_to_be_balanced[_i_exp]
        # Note: For using pytorch, the type of numpy arrays need
        # to be float32.
        i_exp_for_tensor = tmp.astype(np.float32)
        i_exp_tensor = torch.from_numpy(i_exp_for_tensor)
        i_exp_tensor.requires_grad_(True)
        explanatories_tensor_list.append(i_exp_tensor)

    explanatories_shuffled_tensor_list = []
    for _i_data in range(len(explanatories_tensor_list)):
        i_exp_tensor = explanatories_tensor_list[_i_data]
        r = torch.randperm(i_exp_tensor.size()[0])
        i_exp_tensor_suffuled = i_exp_tensor[r, :]
        explanatories_shuffled_tensor_list.append(
          i_exp_tensor_suffuled.detach())
    explanatories_tsr = torch.cat(
        explanatories_tensor_list, dim=1)
    explanatories_shuffled_tsr = torch.cat(
          explanatories_shuffled_tensor_list, dim=1)

    # Create data for pytorch_lightning
    dataset_all_expls_for_nbw = torch.utils.data.TensorDataset(
        explanatories_tsr, explanatories_shuffled_tsr)

    train_dataloaer_nbw = DataLoader(
        dataset_all_expls_for_nbw,
        batch_size=batchsize)
    vaildate_dataloaer_nbw = DataLoader(
        dataset_all_expls_for_nbw,
        batch_size=batchsize)

    test_dataloaer_nbw = DataLoader(
        dataset_all_expls_for_nbw,
        batch_size=batchsize)

    input_dim_nbw = explanatories_tsr.shape[1]
    params_nbw['input_dim'] = input_dim_nbw

    # Build a nbw model
    nbw_models = LightningNbwDense(
                        alpha,
                        params_nbw,
                        learning_rate,
                        previous_nbw_model_filePath_list)
    pytorch_trainer.fit(
        nbw_models,
        train_dataloaer_nbw,
        test_dataloaer_nbw)
    pytorch_trainer.test(nbw_models, test_dataloaer_nbw)

    # Logging
    pytorch_trainer.logger.log_hyperparams(params_nbw)
    test_val_name = 'test_alpha_infomation'
    test_val = pytorch_trainer.callback_metrics[test_val_name].item()
    pytorch_trainer.logger.log_hyperparams(
        params_nbw,
        {test_val_name: test_val})

    adjust_prob_P, adjust_prob_Q = nbw_models(
        explanatories_tsr, explanatories_shuffled_tsr)

    loss = (
        torch.mean(adjust_prob_P)/alpha
        + torch.mean(adjust_prob_Q)/(1 - alpha)
    ).item()
    alpha_infomation = 1/(alpha*(1 - alpha)) - loss
    new_nbw_model = nbw_models.nbw_models_.new_nbw_model_

    return [new_nbw_model, alpha_infomation]


def train_and_enhance_NBW(
            alpha: float,
            explanatories_to_be_balanced: List[Any],
            params_nbw: dict,
            n_enhance_nbw: int,
            learning_rate: float,
            batchsize: int,
            n_epoch: int,
            out_results_dir: str
        ) -> [List[str], float, List[str]]:
    """Built a Neural Balancing Weight (NBW) model (an instance of
       NbwDense class) while tring to enhace the balacing ability of
       their weights.

    This function Creates and trains NBW models “n_enhance_nbw” times.
    The models trained by this function are saved at a location in
    out_results_dir”. Then, This function returns a list of paths of the
    models to be used for esitmating the balancing weights. In addition,
    This function returns an estimate of the alpha-information of the
    distribution balanced by the weights from the models in the list.

    Args:
      alpha: Alpha value for alpha divergence.
      explanatories_to_be_balanced:
          List of 2-dimensional numpy arrays. Each of numpy arrays in
          the list corresponds to an explanatory variable of training
          data, and has rows with the same size and columns with
          various sizes.

      params_nbw:
          Dictionary which contains the folowing parameters for creating
          an instance of NbwDense class.
            * input_dim_: Size of an input layer for each of instances
                          of NbwDense class.
            * hidden_dim_: Size of hedden layers for each of instances of
                           NbwDense class.
            * n_layers: Number of layers for each of instances of
                        NbwDense class.
      n_enhance_nbw: Number of times to enhace the balacing ability of the
                     weights.
      learning_rate: Learning rate for mini-batch stochastic
                     gradient optimization in training a NBW model.
      batchsize: Batchsize for mini-batch stochastic gradient optimization
                 in training a NBW model.
      n_epoch: Number of epochs for mini-batch stochastic gradient
               optimization in training a NBW model.
      out_results_dir: A directory path to output all results of this
                       function, which include pytorch logs for training
                       neural networks.

    Returns:
      filePaths_of_nbw_models_to_use_list: Paths of NGB models (binary
      files of instances of NbwDense class) to be used to estimate the
      balancing weights.
      current_best_alpha_infomation_estimated: An estimate of the
      alpha-information of the distribution balanced by the weights from
      the models "in filePaths_of_nbw_models_to_use_list".
      alpha_infos_of_all_nbw_models: A dictionary from a path of all NBW
      models trained in this function to an estimate of alpha-iformation
      of the distribution balanced by the weights of the models.
    """
    out_log_dir = os.path.join(
        out_results_dir, 'all_logs')
    os.makedirs(out_log_dir, exist_ok=True)
    out_model_dir = os.path.join(out_results_dir, '_nbw_models')
    os.makedirs(out_model_dir, exist_ok=True)

    # Build NBW models n_enhance_nbw times while trying to
    # enhance their balancing.
    # If n_enhance_nbw = 0, only measuring the performance
    # of the balancing is conducted in the next
    # iteration (i.e. _i_nbw = 1).
    alpha_infos_of_all_nbw_models = dict()
    filePaths_of_nbw_models_to_use_list = []
    current_best_alpha_infomation_estimated = np.inf
    previous_nbw_model_filepath = None
    for _i_nbw in range(0, n_enhance_nbw + 1):
        # FilePaths for building a new NBW model
        filePaths_of_nbw_models_for_modeling = \
            filePaths_of_nbw_models_to_use_list.copy()
        if previous_nbw_model_filepath is not None:
            # _i_nbw > 0
            filePaths_of_nbw_models_for_modeling += \
                [previous_nbw_model_filepath]

        # Settings for pytorch lightning trainer
        now = datetime.datetime.now()
        logname_suffix = now.strftime('%Y-%m-%d_%H_%M_%S')
        logname = f'NBW_modeling({_i_nbw})_{logname_suffix}'
        logger = TensorBoardLogger(
            out_log_dir,
            logname,
            default_hp_metric=False)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=2,
            strategy='dp',
            logger=logger,
            max_epochs=n_epoch,
            enable_progress_bar=False)

        new_nbw_model, alpha_infomation_estimated_raw = build_new_nbw_model(
            alpha,
            explanatories_to_be_balanced,
            params_nbw,
            learning_rate,
            batchsize,
            trainer,
            filePaths_of_nbw_models_for_modeling)

        # Save the above results
        out_nbw_mdl_filename = f'nbw_model_{_i_nbw:02}.mi'
        out_nbw_mdl_filepath = os.path.join(
            out_model_dir, out_nbw_mdl_filename)
        torch.save(new_nbw_model.state_dict(),
                   out_nbw_mdl_filepath)
        alpha_infos_of_all_nbw_models[out_nbw_mdl_filepath] = \
            alpha_infomation_estimated_raw

        if _i_nbw > 0:
            alpha_infomation_estimated = abs(alpha_infomation_estimated_raw)
            if alpha_infomation_estimated \
               < current_best_alpha_infomation_estimated*0.99:
                # Here, the balancing of the weights was improved.
                # The current best recorde is updated
                current_best_alpha_infomation_estimated = \
                    alpha_infomation_estimated
                filePaths_of_nbw_models_to_use_list += \
                    [previous_nbw_model_filepath]
        previous_nbw_model_filepath = out_nbw_mdl_filepath

        print('--------------------------------------------------------------')
        print(f'Number of trials to enhacne balancing: {_i_nbw}')
        print(f'Current Best (smallest alpha-information) = ',
              f'{current_best_alpha_infomation_estimated}')
        print('--------------------------------------------------------------')

    return [filePaths_of_nbw_models_to_use_list,
            current_best_alpha_infomation_estimated,
            alpha_infos_of_all_nbw_models]


def estimate_balancing_weights(
      explanatories_for_prediction: List[Any],
      params_nbw: dict,
      filePaths_of_nbw_models_to_use_list: int
  ) -> Any:
    """Estimate balancing weights from the result of the
       "train_and_enhance_NBW" function.

    This is a convinient function to estimate the balancing weights from
    models of "filePaths_of_nbw_models_to_use_list“ (the object of the
    first position of the outputs of the "train_and_enhance_NBW" function).
    The output of this function, i.e. the balancing weights estimated, is
    a 1-dimensional numpy array. The estimated weights are obtained from
    predictions of the models but not normarized. Thus, depending on the
    purpose of your study, you may need to normarize them with the mean
    of themselves.

    Args:
      explanatories_for_prediction: List of 2-dimensional numpy arrays.
      Each of numpy arrays in the list corresponds to an explanatory
      variable of training data, and has rows with the same size and
      columns with various sizes.

      params_nbw:
          Dictionary which contains the folowing parameters for creating
          an instance of NbwDense class.
            * input_dim_: Size of an input layer for each of instances
                          of NbwDense class.
            * hidden_dim_: Size of hedden layers for each of instances of
                            NbwDense class.
            * n_layers: Number of layers for each of instances of
                        NbwDense class.
      filePaths_of_nbw_models_to_use_list:
          Paths of NGB models (binary files of instances of NbwDense
          class) to be used to estimate the balancing  weights.

    Returns:
      result_weights: 1-dimensional numpy array. The balancing weights
                      which are estimated from the NGB models of
                      "filePaths_of_nbw_models_to_use_list".

    Note:
      The output of this function, the estimate of the balancing
      weights, is a simple prediction of the models, and not normarized.
      Thus, depending on the purpose of your study, you may need to
      normarize them with the mean of themselves.
    """
    # The alpha value below (at the first argument in "load_nbw_models"
    # function) is not used, since the models here are used only for
    # estimating the balancing weights.
    nbw_models = load_nbw_models(
          1/2,  # dummy, not used
          params_nbw,
          filePaths_of_nbw_models_to_use_list)

    train_expls_mat = np.concatenate(explanatories_for_prediction, axis=1)
    train_expls_tsr = torch.from_numpy(train_expls_mat)

    esti_nbw_tsr = nbw_models(
        train_expls_tsr, None, 'estimate_only')
    esti_nbw_mat = esti_nbw_tsr.numpy()
    result_weights = esti_nbw_mat.flatten()

    return result_weights
