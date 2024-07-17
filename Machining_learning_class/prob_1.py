from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np


def main():
    iris = datasets.load_iris()

    II_0 = (iris.target == 0)
    II_1 = (iris.target == 1)
    II_2 = (iris.target == 2)

    lower_column_index = 0
    upper_column_index = 2

    fig, ax = plt.subplots()
    ax.scatter(iris.data[II_0, lower_column_index], iris.data[II_0, lower_column_index + 1], color='blue',
               label='setosa')
    ax.scatter(iris.data[II_1, lower_column_index], iris.data[II_1, lower_column_index + 1], color='red',
               label='versicolor')
    ax.scatter(iris.data[II_2, lower_column_index], iris.data[II_2, lower_column_index + 1], color='green',
               label='virginica')
    ax.set_xlabel('petal length')
    ax.set_ylabel('petal width')
    ax.legend()
    fig.show()

    object_knn = KNeighborsClassifier(n_neighbors=3)

    x_train, x_test, y_train, y_test = train_test_split(iris.data[:, lower_column_index:upper_column_index],
                                                        iris.target, test_size=0.2)

    object_knn.fit(x_train, y_train)

    y_predict = object_knn.predict(x_test)

    II_correct = np.where(y_predict == y_test)
    II_incorrect = np.where(y_predict != y_test)

    fig, ax = plt.subplots()
    ax.scatter(iris.data[II_0, lower_column_index], iris.data[II_0, lower_column_index + 1], \
               marker='o', facecolors='none', edgecolors='red', label='setosa')
    ax.scatter(iris.data[II_1, lower_column_index], iris.data[II_1, lower_column_index + 1], \
               marker='o', facecolors='none', edgecolors='green', label='versicolor')
    ax.scatter(iris.data[II_2, lower_column_index], iris.data[II_2, lower_column_index + 1], \
               marker='o', facecolors='none', edgecolors='blue', label='virginica')

    ax.scatter(x_test[II_correct, 0], x_test[II_correct, 1], color='black', marker='*',
               label='correct prediction')
    ax.scatter(x_test[II_incorrect, 0], x_test[II_incorrect, 1], color='magenta', marker='*',
               label='incorrect prediction')
    ax.set_xlabel('petal length')
    ax.set_ylabel('petal width')

    ax.legend()
    fig.show()

    out_file_name = 'KNN-prediction-results.pdf'
    fig.savefig(out_file_name)


if __name__ == '__main__':
    main()
