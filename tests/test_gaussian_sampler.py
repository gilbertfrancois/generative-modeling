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

    fig, axs = plt.subplots(nrows=1, ncols=10, squeeze=True)
    for k in range(10):
        # sample from a given cluster k.
        sample, mean = clf.sample_given_y(k)
        axs[0][k].imshow(sample.reshape(28, 28))
        axs[1][k].imshow(mean.reshape(28, 28))
    plt.show()


if __name__ == "__main__":
    main()
