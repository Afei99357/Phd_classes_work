import pandas as pd
import numpy as np
from sklearn import datasets
from Machining_learning_class import d_similarity_measure

# load iris dataset from sklearn
iris = datasets.load_iris()

# # transfer iris numpy arrays into pandas dataframe
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
iris_df.to_csv('/Users/ericliao/Desktop/phD_courses/2021Spring_class/data_Mining_TA/iris_dataset.csv')

# calculate the euclidean distance between two flowers in iris dataset and add results into a dictionary
euclidean_distance_dic = {}
index = 0
i = 0
for row1 in iris['data']:
    j = 0
    for row2 in iris['data']:
        iris_pair = d_similarity_measure.simimarity_measure(row1, row2)
        euclidean_distance = iris_pair.get_euclidean()
        euclidean_distance_dic[index] = [i, j, euclidean_distance]
        index = index + 1
        j = j + 1
    i = i + 1

# transfer euclidean distance dictionary into a new dictionary with the format we want to output
row_number = 0
output_dict = {}
for row_number in range(len(iris['data'])):
    column_number = 0
    output_dict.setdefault(row_number, [])
    for item in euclidean_distance_dic.items():
        pair_1 = item[1][0]
        pair_2 = item[1][1]
        euclidean_distance = item[1][2]
        if pair_1 == row_number and pair_2 == column_number:
            output_dict[row_number].append(euclidean_distance)
            column_number = column_number + 1
    row_number = row_number + 1

# transfer to pandas dataframe for outputting as csv format file
    output_df = pd.DataFrame.from_dict(output_dict, orient='index')
    output_df.to_csv('/Users/ericliao/Desktop/dict.csv')


