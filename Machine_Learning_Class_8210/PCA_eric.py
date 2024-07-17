# reference : https://www.askpython.com/python/examples/principal-component-analysis
# Author: Eric Liao
# October 2021

"""
How to use this function:
    if run on terminal:
        python3 ..directory_of_the_folder/PCA_eric.py --input_file=here_insert_file_directory --components_number=integer_number_here [--axis=0_or_1_defualt_as 0] [--header=0_or_1_defualt_as 0] [--index=0_or_1_defualt_as 0]
    if run on IDEA:
        In the configuration, at the Parameters section,(Use space to separate each parameter)
        filled with: --input_file=here_insert_file_directory --components_number=integer_number_here [--axis=0_or_1_defualt_as 0] [--header=0_or_1_defualt_as 0] [--index=0_or_1_defualt_as 0]

This function needs 5 inputs:
    1. The dataset (detail information check the help in argument --input_file) **required
    2. The number of components to keep **required
    3. A axis parameter tell the module the structure of your data
        optional, if not provided, then axis=0
    4. A header parameter to tell if the original dataset has header column,
        optional, if not provide, header=0 means no header row
    5. A index parameter to tell if the original dataset has index column,
        optional, if not provide, index=0 means no index column

    (Detailed information of each parameter see the --help in argument in main() function)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skspatial.objects import Line
from skspatial.objects import Point
from skspatial.plotting import plot_2d


def myPCA(X, number_components):
    # # mean centering the data
    # X_mean_center = X - np.mean(X, axis=0)
    #
    # # scaling the data
    # X_standard = X_mean_center / np.std(X_mean_center, axis=0)

    # Calculating the covariance matrix of the mean-centered data, rowvar=False means each row is sample
    # and each column is a feature/variable
    cov_matrix = np.cov(X, rowvar=False)

    # Calculating eigenvalues and eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_index = np.argsort(eigen_values)[::-1]

    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # select the first n eigenvectors, n is desired dimension of our final reduced data (number of components)
    n_components = int(number_components)
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

    # Tansform the data
    scores = np.dot(eigenvector_subset.T, X.T).T

    # calculate how many variance be explained by all eigenvectors
    percent_variance_explained = sorted_eigenvalues / sum(sorted_eigenvalues) * 100

    pca_result = {"scores": scores,
                  "percent_variance_explained": percent_variance_explained,
                  'loadings': sorted_eigenvectors,
                  'eigenvalues': sorted_eigenvalues,
                  'eigenvectors': eigenvector_subset}

    return pca_result


# ## scatter plot
def result_scatter_plot(pca_result):

    # scores plot
    fig, ax = plt.subplots()
    ax.scatter(pca_result['scores'][:, 0], pca_result['scores'][:, 1]/1000, color='blue')
    ax.set_title('scores plot')
    ax.set_xlabel('PC1 (' + str(round(pca_result['percent_variance_explained'][0])) + '%)')
    ax.set_ylabel('PC2 (' + str(round(pca_result['percent_variance_explained'][1])) + '%)')
    ax.set_ylim((-1, 1))

    # scree plot for the first two principal components
    fig, ax = plt.subplots()
    ax.scatter(range(2),
               pca_result['percent_variance_explained'][0:2],
               color='red')
    ax.set_title('scree plot')
    ax.set_xlabel('PC index')
    ax.set_ylabel('percent variance explained')
    ax.set_ylim((-10.0, 110.0))
    for i, label in enumerate(pca_result['percent_variance_explained'][0:2]):
        plt.annotate("{:.2f}".format(label), (i, pca_result['percent_variance_explained'][0:2][i]))

    # loadings plot
    fig, ax = plt.subplots()
    ax.scatter(pca_result['loadings'][:, 0], pca_result['loadings'][:, 1], color='orange')
    ax.set_title('loadings plot')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    for i in range(pca_result['loadings'].shape[0]):
        ax.text(pca_result['loadings'][i, 0], pca_result['loadings'][i, 1], 'x' + str(i + 1))


def main():
    parser = argparse.ArgumentParser(description="Reads factors for each sample and writes them into CSV file")

    # ### total 5 parameters need to specify, first 2 are required, last 3 are optional
    parser.add_argument('--input_file', help='files contains matrix, either csv/txt, separate with spaces',
                        required=True)
    parser.add_argument('--components_number', help='number of principal components to keep', required=True)
    parser.add_argument('--axis', nargs='?', const=1, type=int, default=0,
                        help='axis value will be integer number(0 or 1), default is 0, '
                             'if axis = 0, the structure of dataset is rows as samples, columns as features/variables; '
                             'if axis = 1, the structure of dataset is rows as features/variables, columns as samples.'
                             'The example of dataset with axis=0 and axis=1 is in the folder called data_file_example')
    parser.add_argument('--header', nargs='?', const=1, type=int, default=0,
                        help='header value will be integer number(0 or 1), default is 0'
                             'if header = 0, the first row is not column name; '
                             'if header = 1, the first row is column name.')
    parser.add_argument('--index', nargs='?', const=1, type=int, default=0,
                        help='index value will be integer number(0 or 1), default is 0'
                             'if index = 0, the first column is not index column; '
                             'if index = 1, the first column is index column.')

    args = parser.parse_args()

    input_file = args.input_file
    pc_number = args.components_number
    axis = args.axis
    header = args.header
    index = args.index

    # #check if the data table has column name and index column
    if header == 0 and index == 0:
        original_df = pd.read_csv(input_file)
    if header == 1 and index == 1:
        original_df = pd.read_csv(input_file, header=0, index_col=0)
    if header == 0 and index == 1:
        original_df = pd.read_csv(input_file, index_col=0)
    if header == 1 and index == 0:
        original_df = pd.read_csv(input_file, header=0)

    data_df = original_df.copy()

    # transfer dataframe to numpy array
    X = data_df.to_numpy()

    # original data table structure is row as features/variables and columns as samples
    if axis == 1:
        X = X.T

    # do PCA
    # todo: change the data columns manually
    pca_result = myPCA(X[:, 0:2], pc_number)

    print(pca_result['eigenvalues'])

    print(pca_result['eigenvectors'])
    #
    # pca_result = PCA(n_components=2)
    # pca_result.fit(X[:, 0:2])
    # plot results
    result_scatter_plot(pca_result)

    # v1 = original_df['V1'].to_numpy()
    # v2 = original_df['V2'].to_numpy()
    # label = original_df['label'].to_numpy()
    #
    # plt.scatter(v1, v2, c=label, edgecolor="none", alpha=0.8)
    #
    # x_values = [0, pca_result["loadings"][0][0] * 45]
    # y_values = [0, pca_result["loadings"][1][0] * 45]
    # plt.plot(x_values, y_values)

    # x_values = [1, 0]
    # y_values = [-0.71381103 * 40, 0.70033836 * 40]
    # plt.plot(x_values, y_values)

    # # plot pc1 projection
    # fig, ax = plt.subplots()
    # ax.scatter(pca_result['scores'][:, 0], pca_result['scores'][:, 1], color='blue')
    # ax.set_title('scores plot')
    # ax.set_xlabel('PC1 (' + str(round(pca_result['percent_variance_explained'][0])) + '%)')
    # ax.set_ylabel('PC2 (' + str(round(pca_result['percent_variance_explained'][1])) + '%)')
    # ax.set_ylim((2 * min(pca_result['scores'][:, 1]), max(X[:, 1])))

    plt.show()


if __name__ == '__main__':
    main()
