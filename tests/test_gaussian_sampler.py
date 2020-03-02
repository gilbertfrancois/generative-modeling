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

import pandas as pd
import matplotlib.pyplot as plt
from bayes_classifier.bayes_classifier_gaussian import BayesClassifierGaussian


def main():

    # Load data
    data = pd.read_csv("../data/train.csv")
    Y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    X = X / 255

    # Create the generative model.
    clf = BayesClassifierGaussian()
    clf.fit(X, Y)

    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for k in range(10):
        # sample from a given cluster k.
        res = clf.sample_given_y(k)
        axs[0, k].imshow(res["sample"].reshape(28, 28))
        axs[1, k].imshow(res["mean"].reshape(28, 28))
        axs[0, k].axis("off")
        axs[1, k].axis("off")
    fig.suptitle("Sampling $P(x|y)$ from a single Gaussian per class")
    plt.show()

    # Sample from a random class.
    res = clf.sample()
    fig, axs = plt.subplots(2, 1, figsize=(4, 8))
    axs[0].imshow(res["sample"].reshape(28, 28))
    axs[1].imshow(res["mean"].reshape(28, 28))
    axs[0].set_title(res["class"])
    plt.show()


if __name__ == "__main__":
    main()
