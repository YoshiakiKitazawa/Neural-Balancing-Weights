# Library for Generalized Balancing Weights via Deep Neural Networks

This repository contains a library[^1][^2] for estimating Neural Balancing Weights (NBW)
proposed in our paper ― [Generalized Balancing Weights via Deep Neural
Networks](https://arxiv.org/abs/2211.07533).
For reproducing results of experiments in the paper, run
[the original code](./reproduction/HowToReproduce.md) used in the paper.

[^1]: This library is made from [the original code](./reproduction/HowToReproduce.md) used in the paper for the purpose of
poviding tools to easely estimate balancing weights for causal inference. 

[^2]: I will not update this repository. However, I hope some code here give everyone a bit of help for his/her research.

**Preparation**
---
For using this library, one may need to make the virtual environment and install all the requirements using Anaconda Distribution.
```
conda env create --name nbwenv python=3.9 -f requirements.yaml
source activate nbwenv
```

**Usage**
---
We now estimate the balancing weights with a dense neural network with 
20 units × 8 hidden layer in the following settings.

### Settings for building NBW models
#### Settings for mini-batch stochastic gradient optimization in training: 
* Learning rate of SGD is 0.001.
* Batch size of SGD is 2500.
* Number of epochs of SGD is 1000.

#### Settings for estimateing the balancing weights:
* α is 1/2.
* Number of trial to enhance their balancing is 5.

#### Output directory for all resuls of building the NBW models: 
* directory to output results is ```'[current directory]/out/'```

### Training data
Training data needs to be a list of 2-dimensional numpy arrays.
Each of numpy arrays in the list corresponds to an explanatory variable of training data,
and has rows with the same size and columns with various sizes.



 

### Building NBW models 
A sample code of estimating the weights in the above setting is as follows.
Here, ```train_expls``` is  training data (i.e. a list of 2-dimensional numpy arrays).
```
# Neural network structure 
params_nn = {
        'hidden_dim': 20
        'n_layers': 8}

# Build NBW models while tring to enhance the balancing ability of the 
# weights 5 times
filePaths_of_nbw_models_to_use_list, \
    final_alpha_infomation_estimated, \
    alpha_infos_of_all_nbw_models = train_and_enhance_NBW(
        1/2,            # α
        train_expls,    # learning data (list of numpy array) 
        params_nn,      
        5,              # number of trial to enhance the balancing ability
                        # of the weights
        0.001,          # learning rate of SGD
        2500,           # batch size of SGD
        1000,           # number of epochs of SGD
        './out'         # dirctory to output all resutls
  )
```
The details of the outputs in the above code are as follows:
* ```filePaths_of_nbw_models_to_use_list``` 
    - This is a list of paths of NBW models to be used to estimate the balancing weights.
    - All models built in the ```train_and_enhance_NBW``` function are saved
      at ```'./out/_nbw_models/'``` in binary format. This list only contains paths of 
      the models who enchanced the balancing ability of the weights.
* ```current_best_alpha_infomation_estimated```
    - An estimate of the alpha-information of the distribution balanced by the weights obtained from the 
      all models of ```filePaths_of_nbw_models_to_use_list```.
* ```alpha_infos_of_all_nbw_models```
    - A dictionary from a path of all NBW models trained to an estimate of alpha-iformation of 
      the distribution balanced by the weights of the models.

### Estimating balancing weights
A sample code of estimating the weights from the above results is as follows:
```
balancing_weights_estimated = estimate_balancing_weights(
      train_expls,
      params_nbw,
      filePaths_of_nbw_models_to_use_list)
```

### Using the balancing weights
A sample code for modeling Linear Regression with the balancing weights obtained above is as follows.

```
from sklearn.linear_model import LinearRegression

# Bind arrays of train_expls to one array along the columns 
# for using LinearRegression.fit() function
expls_to_fit = np.concatenate(train_expls, axis=1)

# Built a model of Linear Regression.
# Here, respons_to_fit is a response variable, which is a 1-D numpy
# array with the same length as the row-size of expls_to_fit.
model = LinearRegression().fit(
    expls_to_fit, 
    respons_to_fit,
    sample_weight=balancing_weights_estimated)
```

**Example**
---
For examples of a full code using the above functions, see the following scripts.
*  [Run_single_simulation_Experiment1.py](./Run_single_simulation_Experiment1.py)
    - A script using the above functions, which estimates the (heterogeneous) average causal effect using Gradient Boosting Tree Algorithm in the setting of Experiment 1 in our paper.
*  [Run_single_simulation_Experiment1.py](./Run_single_simulation_Experiment2.py)
    - A script using the above functions, which estimates the (heterogeneous) average causal effect using Gradient Boosting Tree Algorithm in the setting of Experiment 2 in our paper.