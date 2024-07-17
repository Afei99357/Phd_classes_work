import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn.cross_decomposition import PLSRegression
import os
from Lab_work.PLS_DA_for_2_components.vips import vips
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

home = os.path.dirname(__file__) + "/predict-west-nile-virus/west_nile/"

# Load dataset
train = pd.read_csv(home + "input/train.csv")
test = pd.read_csv(home + "input/test.csv")
sample = pd.read_csv(home + "input/sampleSubmission.csv")
weather = pd.read_csv(home + "input/weather.csv")

# Get labels
labels = train.WnvPresent.values
# test_labels = sample.WnvPresent.values
# Not using codesum for this benchmark
weather = weather.drop("CodeSum", axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather["Station"] == 1]
weather_stn2 = weather[weather["Station"] == 2]
weather_stn1 = weather_stn1.drop("Station", axis=1)
weather_stn2 = weather_stn2.drop("Station", axis=1)
weather = weather_stn1.merge(weather_stn2, on="Date")

# replace some missing values and T with -1
weather = weather.replace("M", -1)
weather = weather.replace("-", -1)
weather = weather.replace("T", -1)
weather = weather.replace(" T", -1)
weather = weather.replace("  T", -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split("-")[1]


def create_day(x):
    return x.split("-")[2]


train["month"] = train.Date.apply(create_month)
train["day"] = train.Date.apply(create_day)
test["month"] = test.Date.apply(create_month)
test["day"] = test.Date.apply(create_day)

# Add integer latitude/longitude columns
train["Lat_int"] = train.Latitude.apply(int)
train["Long_int"] = train.Longitude.apply(int)
test["Lat_int"] = test.Latitude.apply(int)
test["Long_int"] = test.Longitude.apply(int)

# drop address columns
train = train.drop(
    ["Address", "AddressNumberAndStreet", "WnvPresent", "NumMosquitos"], axis=1
)
test = test.drop(["Id", "Address", "AddressNumberAndStreet"], axis=1)

# Merge with weather data
train = train.merge(weather, on="Date")
test = test.merge(weather, on="Date")
train = train.drop(["Date"], axis=1)
test = test.drop(["Date"], axis=1)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train["Species"].values) + list(test["Species"].values))
train["Species"] = lbl.transform(train["Species"].values)
test["Species"] = lbl.transform(test["Species"].values)

lbl.fit(list(train["Street"].values) + list(test["Street"].values))
train["Street"] = lbl.transform(train["Street"].values)
test["Street"] = lbl.transform(test["Street"].values)

lbl.fit(list(train["Trap"].values) + list(test["Trap"].values))
train["Trap"] = lbl.transform(train["Trap"].values)
test["Trap"] = lbl.transform(test["Trap"].values)

# drop columns with -1s
# train = train.ix[:,(train != -1).any(axis=0)]
# test = test.ix[:,(test != -1).any(axis=0)]

# Random Forest Classifier
# clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1)

# clf = PLSRegression(n_components=2, scale=False)
# clf.fit(train, labels)

scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train))

# ###################### PLS-DA #####################################
# Q2_list_1 = []
# Q2_list_2 = []
# for i in range(1000):
#     X_train, X_test, y_train, y_test = train_test_split(train,
#                                                         labels,
#                                                         test_size=0.30,
#                                                         shuffle=True)
#
#     x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X_train,
#                                                                 y_train,
#                                                                 test_size=0.30,
#                                                                 shuffle=True)
#
#     plsr = PLSRegression(n_components=2, scale=False)
#     plsr.fit(x_train_2, y_train_2)
#     y_predict_2 = plsr.predict(x_test_2)
#
#     q2 = metrics.r2_score(y_test_2, y_predict_2)
#
#     Q2_list_1.append(q2)
#
#     vip_value = vips.vips(plsr)
#
#
#     selected_test_df = X_test.iloc[:, vip_value > 1]
#
#     x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(X_test.loc[:, vip_value > 1],
#                                                                 y_test,
#                                                                 test_size=0.30,
#                                                                 shuffle=True)
#
#     plsr_2 = PLSRegression(n_components=2, scale=False)
#     plsr_2.fit(x_train_3, y_train_3)
#     y_predict_3 = plsr_2.predict(x_test_3)
#
#     q2_2 = metrics.r2_score(y_test_3, y_predict_3)
#     Q2_list_2.append(q2_2)
#     print(i)
#
# plt.hist(Q2_list_1, 100, label='first Q2')
#
# plt.figure()
# plt.hist(Q2_list_2, 100, label='second Q2')
#
# plt.legend()
# plt.show()
# ###################################################################

# ########################### SVC ####################################
# X_train, X_test, y_train, y_test = train_test_split(train,
#                                                     labels,
#                                                     test_size=0.30,
#                                                     shuffle=True)
# svc = SVC(gamma='auto')
# svc.fit(X_train, y_train)
# y_predict = svc.predict(X_test)
#
# q2 = metrics.r2_score(y_test, y_predict)
# f1 = f1_score(y_test, y_predict)
#
# print("Q2 is ", q2)
# print("F1 score is ", f1)
# ######################################################################

########################## RF ######################################
### Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(train,
                                                    labels,
                                                    test_size=0.30,
                                                    shuffle=True)

clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

q2 = metrics.r2_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

print("Q2 is ", q2)
print("F1 score is ", f1)
#####################################################################

########################### Random Kitchen-sink  #####################
# X_train, X_test, y_train, y_test = train_test_split(
#     train, labels, test_size=0.30, shuffle=True
# )
# from sklearn.kernel_approximation import Nystroem
#
# tran = Nystroem(n_components=10)
# clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=3)
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
#
# q2 = metrics.r2_score(y_test, y_predict)
# f1 = f1_score(y_test, y_predict)
#
# print("Q2 is ", q2)
# print("F1 score is ", f1)
######################################################################
