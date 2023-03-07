# Copyright 2020 Gilbert François Duivesteijn

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.stats import multivariate_normal


def clamp(x):
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x


class BayesClassifierGaussian:
    def __init__(self):
        """Constructor"""
        # Models per class
        self.gaussian_list = []
        # Number of classes
        self.K = 0
        # probability of a given class
        self.Py = np.zeros(self.K)

    def fit(self, X, Y):
        """Fit the model for the given training data.

        Parameters
        ----------
        X: np.array
            Data with shape (N, M) where N is number of samples and M is number
            of properties.
        Y: np.array
            Labels with shape (N,) where the labels are numbered like 0..K-1
        """
        self.K = len(set(Y))
        self.Py = np.zeros(self.K)
        for k in range(self.K):
            Xk = X[Y == k]
            self.Py[k] = len(Xk)
            mu = np.mean(Xk, axis=0)
            sigma = np.cov(Xk.T)
            g = {"mu": mu, "sigma": sigma}
            self.gaussian_list.append(g)
        self.Py = self.Py / self.Py.sum()

    def sample_given_y(self, y):
        """Select a gaussian for class y and sample from it.

        Parameters
        ----------
        y: int
            Class y

        Returns
        -------
        (np.array, np.array)
            Tuple of the sample and the mean of the class
        """
        g = self.gaussian_list[y]
        sample = clamp(multivariate_normal.rvs(mean=g["mu"], cov=g["sigma"]))
        mean = g["mu"]
        return {"sample": sample, "mean": mean, "class": y}

    def sample(self):
        """Pick a random class and sample from it.

        Returns
        -------
        (np.array, np.array)
            Tuple of the sample and the mean of the class
        """
        y = np.random.choice(self.K, p=self.Py)
        return self.sample_given_y(y)
