from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# load iris dataset from sklearn
iris = datasets.load_iris()

# # transfer iris numpy arrays into pandas dataframe
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
# iris_df.to_csv('/Users/ericliao/Desktop/phD_courses/2021Spring_class/data_Mining_TA/iris_dataset.csv')

X = iris_df.iloc[50:151, 0:4]
y = iris_df.iloc[50:151, -1:]

repeat_time = 20
avg_error_rate = []
lda_model = LinearDiscriminantAnalysis()

for i in range(repeat_time):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # use trainning data Fit model
    # .values will give the values in an array. (shape: (n,1))
    # .ravel() will convert that array shape to ndarray (n, )
    lda_model.fit(X_train, y_train.values.ravel())

    # predict using 10% data
    predict_result = lda_model.predict(X_test)
    y_test_np_array = y_test.to_numpy()
    # reshape the 1d array with 1 row, 10 columns to 10 rows, 1 column array
    avg_error_rate.append(np.mean(np.sum(predict_result.ravel() == y_test_np_array)))

plt.plot(range(repeat_time), avg_error_rate)
plt.title('Linear Discriminant Analysis Classifier')
plt.xlabel('prediction indices')
plt.ylabel('error rate')
plt.show()

