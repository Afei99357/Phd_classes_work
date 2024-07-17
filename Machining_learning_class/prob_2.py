from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def main():
    iris = datasets.load_iris()

    number_of_tests = 10
    prediction_accuracy = np.zeros(number_of_tests)
    for i in range(number_of_tests):
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

        object_dt = DecisionTreeClassifier(max_depth=4)
        object_dt.fit(x_train, y_train)
        # Dots per inches (dpi) determines how many pixels the figure comprises
        fig, ax = plt.subplots(dpi=600)
        tree.plot_tree(object_dt, feature_names=iris.feature_names, \
                       class_names=iris.target_names, filled=True)

        out_file_name = '/Users/ericliao/Desktop/phD_courses/2021Spring_class/data_Mining_TA/assignment_2/iris-decision-tree_' + str(i) + '.png'
        fig.savefig(out_file_name)

        y_prediction = object_dt.predict(x_test)

        prediction_accuracy[i] = metrics.accuracy_score(y_test, y_prediction)

    print(prediction_accuracy)
    print(np.mean(prediction_accuracy))


if __name__ == '__main__':
    main()
