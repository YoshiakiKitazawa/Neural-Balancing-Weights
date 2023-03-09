# Reproducing Experiments in Our Paper

This document discribes how to reproduce the results of Experiment 1 and Experiment 2 
in our paper ― [Generalized Balancing Weights via Deep Neural Networks](https://arxiv.org/abs/2211.07533).


**Installation**
---
First one needs to make the virtual environment and install all the requirements using Anaconda Distribution.
```
conda env create --name nbwenv python=3.9 -f requirements.yaml
source activate nbwenv
```

**Usage**
---

```
Usage: python experiment1.py N [Options]
Usage: python experiment2.py N [Options]


Positional arguments:
  N                   Number of simulations. For setting the value, please read 
                      the below note.

Options:
  -output_dir  dir    The directory path to output results. The default 
                      directory is the current directoy. 

  -alpha a            alpha-divegence to use for estimating desity ratios.
                      The default is a=0.5.
  -n_layer_nn L       Number of layers of a neural network. The default is 
                      L=8.
  -n_units_hidden H   Number of units of hidden layer. The default is H=20.
                      Each of the simulations uses a neural network that has 
                      (L+1) dense layers with H hidden units.
  -n_enhance_nbw U    Number of times to enhance NBW. The default is U=5.
  -n_epochs E         Number of Epochs. The default is E=1000.
  -batchsize B        Sample size of mini-batch. The default is B=2500.
  -learning_rate l    Learning rate for training neural networks. The 
                      default is l=0.0001.
```
## Note for Running Time:
 Each simulation takes about 30 minutes, because the simulation tries to enhance the weights 5 times in the defalut setting. For the first run, please use a small number of simulations (ex. N=2).

---
**Example usage**
---
Exp.1) Run 3 simulations of Experiment 1.

```
python experiment1.py 3
```
Exp.2) Run 100 simulations of Experiment 2 and output the results to './out/'.

```
python experiment2.py 100 -output_dir './out/'
```
---

**Output**
---
The output directory structure for the results of the simulations is as follows:
Here, [Experiment_name] is "Experiment1" or "Experiment2", depending on whether experiment1.py or experiment2.py is executed. [output_dir] is the value of "-output_dir" option argument (default='./').


```
[output_dir]/
├── [Experiment_name]/
        ├── all_logs/
        └── all_results/
                ├── [Experiment_name]_all.csv
                ├── [Experiment_name]_summary.csv
                └── all_results_*.pickle
```
### The details of output

* all_logs/
  - The logs of pytorch for each simulation.

* all_results/[Experiment_name]_all.csv

  - All the MSE's for the algorithms in the numerical experiment. The results are shown in 
      a table: each of the columns coressponds to one of the algorithms; each of the rows, one of the simulations.
      
* all_results/[Experiment_name]_summary.csv

    - The means and standard deviations obtained from [Experiment_name]_all.csv. 

* all_results/all_results_*.pickle 

    - The binary results of [Experiment_name]_aall.csv splitted by each result of the algorithms.


