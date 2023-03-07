import time

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn


def gpu_exists():
    try:
        mx.nd.zeros((1,), ctx=mx.gpu(0))
    except:
        return False
    return True


data_ctx = mx.cpu()

if gpu_exists():
    print("Using GPU for model context.")
    model_ctx = mx.gpu(0)
else:
    print("Using CPU for model context.")
    model_ctx = mx.cpu(0)

mx.random.seed(1)

# %%
# Load MNIST

mnist_data = mx.test_utils.get_mnist()

n_samples = 10


def show_samples(n_samples, mnist_data):
    idx_list = np.random.choice(len(mnist_data["train_data"]), n_samples)
    fig, axs = plt.subplots(1, n_samples)
    for i, j in enumerate(idx_list):
        axs[i].imshow(mnist_data["train_data"][j][0], cmap="Greys")
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])
    plt.show()


train_data = np.reshape(mnist_data["train_data"], (-1, 28 * 28))
test_data = np.reshape(mnist_data["test_data"], (-1, 28 * 28))


class VAE(gluon.HybridBlock):
    def __init__(
        self, n_hidden=400, n_latent=2, n_layers=1, n_output=768, batch_size=100
    ):
        # TODO Continue implementation
        pass
