import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None
        self.eigenvalues = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * mean_diff.dot(mean_diff.T)

        A = np.linalg.inv(SW).dot(SB)

        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]
        self.eigenvalues = eigenvalues[0:self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)


def main():
    data = pd.read_csv('/Users/yliao13/PycharmProjects/phd_class/Machine_Learning_Class_8210/data_files/iris_dataset.csv', header=0, index_col=0)
    X, y = data.iloc[:, 0:4].to_numpy(), data['target'].to_numpy()

    # Project the data onto the 2 primary linear discriminants
    lda = LDA(2)

    iris = load_iris()
    lda.fit(X, y)
    X_projected = lda.transform(X)

    # # sklearn lda
    # clf = LinearDiscriminantAnalysis()
    # clf.fit(X, y)
    # clf_projected = clf.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    plt.scatter(x1, x2/1000, c=y, edgecolor="none", alpha=0.8)
    plt.ylim(-1, 1)

    plt.xlabel("LD 1  " + str(round((lda.eigenvalues[0] * 100) /(lda.eigenvalues[0] + lda.eigenvalues[1]))) + "%")
    plt.ylabel("LD 2 " + str(round((lda.eigenvalues[1] * 100)/(lda.eigenvalues[0] + lda.eigenvalues[1]))) + "%")

    plt.title("Eric's LDA")
    plt.show()


if __name__ == "__main__":
    main()
