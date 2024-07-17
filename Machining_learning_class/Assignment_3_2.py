from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load iris dataset from sklearn
iris = datasets.load_iris()

# # transfer iris numpy arrays into pandas dataframe
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# # transfer iris numpy arrays into pandas dataframe
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
# iris_df.to_csv('/Users/ericliao/Desktop/phD_courses/2021Spring_class/data_Mining_TA/iris_dataset.csv')

X = iris_df.iloc[50:151, 0:4]
y = iris_df.iloc[50:151, -1:]

repeat_time = 20
avg_error_rate = []
gnb_model = GaussianNB()

for i in range(repeat_time):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # use trainning data Fit model
    gnb_model.fit(X_train, y_train.values.ravel())
    # predict using 10% data
    predict_result = gnb_model.predict(X_test)
    y_test_np_array = y_test.to_numpy()
    avg_error_rate.append(np.mean(np.sum(predict_result == y_test_np_array)))

plt.plot(range(repeat_time), avg_error_rate)
plt.title('Naive Bayes Classifier')
plt.xlabel('prediction indices')
plt.ylabel('error rate')
plt.show()

# part 2:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

gnb_model.fit(X_train, y_train.values.ravel())
y_pred_proba = gnb_model.predict_proba(X_test)

# y_pred_proba_result = np.array2string(y_pred_proba, formatter={"float": lambda y_pred_proba: "%.2f" % y_pred_proba})

print(y_pred_proba)