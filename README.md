# Sampling using Bayesian Classifier

## Abstract

To do...


## Setup

Go to the Kaggle website. Create an account, if you have not done that yet,
and download the MNIST handwritten digits set: `train.csv` and `test.csv`.
Place the 2 files in the `data/` folder.

When using poetry as venv manager, just type `poetry install`. If you prefer
pipenv or plain old pip, look in the `pyproject.toml` file for the necessary
libraries and install them in your python environment.

To run the examples:

```sh
# When using poetry venv
poetry shell

# Set the python path
export PYTHONPATH=`pwd`/src

# Go to the tests folder and run the examples.
cd tests
python test_....py
```


## Sampling

Now we sample from the model $P(x|y)$. When using the Bayesian Classifier with single
Gaussian modeling, it draws samples from a single Gaussian per class. The drawn 
samples don't look sharp. The path is:
```
Y -> X
```
<img src="./data/images/samples_gaussian.png">

When using the Bayesian Classifier with Gaussian Mixture Models, it draws samples
from one of the models in the given class. So, the path is from the class, via
the latent space, to a sample:
```
Y -> Z -> X
```
<img src="./data/images/samples_gmm.png">

