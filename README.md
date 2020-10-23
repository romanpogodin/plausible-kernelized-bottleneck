# Plausible kernelized bottleneck
Implementation of "Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks" by R. Pogodin and P. E. Latham (https://arxiv.org/abs/2006.07123)

### Experiments
Scripts with reported experiments are stored in `./experiments_scripts/`.
The `_grid.sh` files run a loop over all setups, calling `_single.sh` that 
sets the hyperparameters and executes `experiments.py`.
 
`experiment.py` runs a single experiments with the given command line arguments;
 run `python3 experiments.py --help`  for the list of arguments
  (or see `utils.py`, function `parse_arguments`).
  
#### 3-layer MLP with 1024 neurons in each layer:
`./experiments_scripts/run_mlp_experiments_grid.sh`

Mean test accuracy over 5 trials (first row: method; 
cossim: cosine similarity kernel;
Gaussian: Gaussian kernel; 
last layer: training of the last layer only;
second row: additional modification of the method
grp+div: grouping with divisive normalization;
 grp: grouping without divisive normalization):

|         | backprop |         | last layer |         | cossim |      |         | Gaussian |      |         |
|---------|:--------:|:-------:|:----------:|:-------:|:------:|:----:|:-------:|:--------:|:----:|:-------:|
|         |          | grp+div |            | grp+div |        |  grp | grp+div |          |  grp | grp+div |
| MNIST   |   98.6   |   98.4  |    92.0    |   95.4  |  94.9  | 95.8 |   96.3  |   94.6   | 98.4 |   98.1  |
| fMNIST  |   90.2   |   90.8  |    83.3    |   85.7  |  86.3  | 88.7 |   88.1  |   86.5   | 88.6 |   88.8  |
| kMNIST  |   93.4   |   93.5  |    71.2    |   78.2  |  80.4  | 86.2 |   87.2  |   80.2   | 92.7 |   91.1  |
| CIFAR10 |   60.0   |   60.3  |    39.2    |   38.0  |  51.1  | 52.5 |   47.6  |   41.4   | 48.4 |   46.4  |

#### VGG-like network:
`./experiments_scripts/run_vgg_sgd_experiments_grid.sh`

`./experiments_scripts/run_vgg_adam_experiments_grid.sh`

Mean test accuracy on CIFAR10 over 5 runs for a 7-layer conv nets (1x and 2x wide).
Cossim: cosine similarity; divnorm: divisive normalization; bn: batchnorm.
Empty entries: experiments for which we didn't find a satisfying set of parameters
due to instabilities in the methods.

|                                 | backprop |      | pHSIC: cossim |         | pHSIC: Gaussian |         |
|---------------------------------|:--------:|:----:|:-------------:|:-------:|:---------------:|:-------:|
|                                 |          |  div |      grp      | grp+div |       grp       | grp+div |
| 1x wide net + SGD               |   91.0   | 91.0 |      88.8     |   89.8  |                 |   86.2  |
| 2x wide net + SGD               |   91.9   | 90.9 |      89.4     |   91.3  |                 |   90.4  |
| 1x wide net + AdamW + batchnorm |   94.1   | 94.3 |      91.3     |   90.1  |       89.9      |   89.4  |
| 2x wide net + AdamW + batchnorm |   94.3   | 94.5 |      91.9     |   91.0  |       91.0      |   91.2  |

Mean test accuracy on CIFAR10 over 5 runs for a 7-layer conv nets (1x and 2x wide).
 FA: feedback alignment; sign sym.: sign symmetry;
  layer class.: layer-wise classification; divnorm: divisive normalization;
   bn: batchnorm. Empty entries: experiments for which we didn't find a satisfying 
   set of parameters due to instabilities in the methods.

|                             |  FA  | sign sym. | layer class. |      |
|-----------------------------|:----:|:---------:|:------------:|:----:|
|                             |      |           |              |  +FA |
| 1x wide net + SGD           |      |           |     90.0     |      |
| 2x wide net + SGD           |      |           |     90.3     |      |
| 1x wide net + SGD + divnorm | 80.4 |    89.5   |     90.5     | 81.0 |
| 2x wide net + SGD + divnorm | 80.6 |    91.3   |     91.3     | 81.2 |
| 1x wide net + AdamW + bn    | 82.4 |    93.6   |     92.1     | 90.3 |
| 2x wide net + AdamW + bn    | 81.6 |    93.9   |     92.1     | 91.1 |

### Requirements
The code was tested with the following setup:
```
OS: Debian GNU/Linux 9.12 (stretch)
CUDA: Driver Version: 418.87.01, CUDA Version: 10.1
Python 3.7.6
numpy 1.18.1
torch 1.6.0+cu101
torchvision 0.7.0+cu101
```
