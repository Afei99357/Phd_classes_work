import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC
from tensorflow_addons import metrics
import seaborn as sns

home = os.path.dirname(__file__) + "/predict-west-nile-virus/west_nile/"

# Load dataset
train = pd.read_csv(home + "input/train.csv")
train_original = train.copy()
sample = pd.read_csv(home + "input/sampleSubmission.csv")
weather = pd.read_csv(home + "input/weather.csv")

# Get labels
labels = train.WnvPresent.values
test_labels = sample.WnvPresent.values
train = train.drop(["Address", "Block", "Street", "AddressNumberAndStreet", "AddressAccuracy", "WnvPresent"], axis=1)

# convert Dates by plotting them to a circle to see the seasonal affect
train['year_sin'] = np.sin(train['Date'].to_numpy().astype("datetime64[D]").astype(float) * np.pi / 365)
train['year_cos'] = np.cos(train['Date'].to_numpy().astype("datetime64[D]").astype(float) * np.pi / 365)
## Not using codesum for this benchmark
weather = weather.drop("CodeSum", axis=1)

## Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather["Station"] == 1]
weather_stn2 = weather[weather["Station"] == 2]
weather_stn1 = weather_stn1.drop("Station", axis=1)
weather_stn2 = weather_stn2.drop("Station", axis=1)
weather = weather_stn1.merge(weather_stn2, on="Date")

## replace some missing values and T with -1
weather = weather.replace("M", -1)
weather = weather.replace("-", -1)
weather = weather.replace("T", -1)
weather = weather.replace(" T", -1)
weather = weather.replace("  T", -1)

## Merge with weather data
train = train.merge(weather, on="Date")
train = train.drop(["Date"], axis=1)

## Convert categorical data Species to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(train["Species"].values)
train["Species"] = lbl.transform(train["Species"].values)

## Convert categorical data Trap to numbers
lbl.fit(train["Trap"].values)
train["Trap"] = lbl.transform(train["Trap"].values)

## train, test data split
X_train, X_test, y_train, y_test = train_test_split(
    train, labels, test_size=0.30, shuffle=True
)

## get categorical feature 'Species' and "Trap"
species_train = X_train.pop("Species").to_numpy()
species_test = X_test.pop("Species").to_numpy()
trap_train = X_train.pop("Trap").to_numpy()
trap_test = X_test.pop("Trap").to_numpy()

## preprocessing
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

############################ Neural Network ##########################



## prepare for neural network
input_non_categorical_shape = (X_train.shape[1],)

## Calculate the number of unique values in the categorical features for embedding input layer
input_categorical_species_shape = np.unique(np.concatenate((species_train, species_test), axis=0)).size
input_categorical_trap_shape = np.unique(np.concatenate((trap_train, trap_test), axis=0)).size

## Define the input layers for species
cat_species_input = pipe1 = layers.Input(shape=(1,), name="cat_species")
## embedding layer for Species
pipe1 = layers.Embedding(input_categorical_species_shape, 256)(pipe1)
pipe1 = tf.reshape(pipe1, [-1, 256])

## Define the input layers for trap
cat_trap_input = pipe2 = layers.Input(shape=(1,), name="cat_trap")
## embedding layer for Trap
pipe2 = layers.Embedding(input_categorical_trap_shape, 256)(pipe1)
pipe2 = tf.reshape(pipe1, [-1, 256])

## Define the input layers for non-categorical features and add batch normalization
noncat_input = pipe = layers.Input(shape=input_non_categorical_shape, name="noncat")
pipe = layers.BatchNormalization()(pipe)
## Create the hidden layer the rest of the neural network layers
pipe = layers.Dense(256, activation='relu')(pipe)
pipe = pipe + pipe1 + pipe2
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.3)(pipe)
pipe = layers.Dense(256, activation='relu')(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.3)(pipe)
pipe = layers.Dense(1, activation='sigmoid')(pipe)

# This uses layers with shared weights, which is different but useful
# input = pipe = layers.Input(shape=(14,))
# pipe = layers.BatchNormalization(input_shape=input_shape)(pipe)
# d1 = layers.Dense(256, activation='relu')
# d2 = layers.BatchNormalization()
# d3 = layers.Dropout(0.3)
# pipe = d3(d2(d1(pipe)))
# pipe = d3(d2(d1(pipe)))
# pipe = layers.Dense(1, activation='sigmoid')(pipe)

model = keras.models.Model(inputs=[cat_species_input, cat_trap_input, noncat_input], outputs=[pipe])

# create an instance of the AUC metric
auc_metric = AUC()

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['binary_accuracy', metrics.F1Score(1), auc_metric],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

input_train_dict = {"cat_species": species_train, "cat_trap": trap_train, "noncat": X_train}
input_test_dict = {"cat_species": species_test, "cat_trap": trap_test, "noncat": X_test}

# add class weights for imbalanced class data
class_weight = {0: 1, 1: 100}

# train the model and add class weights for imbalance
history = model.fit(
    input_train_dict, y_train.reshape((-1, 1)),
    validation_data=(input_test_dict, y_test.reshape((-1, 1))),
    batch_size=512,
    epochs=1000,
    class_weight=class_weight
    # callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
history_df.loc[:, ['auc', 'val_auc']].plot(title="AUC")

plt.show()
######################################################################

########################## data visualization ########################
# plot the distribution of the target variable
sns.countplot(x='WnvPresent', data=train_original)
plt.show()

# pair plot for all features
train['WnvPresent'] = train_original['WnvPresent']
sns.pairplot(train, hue='WnvPresent')
plt.show()
