#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import layers
import numpy as np
import code

class Dataset:
    def __init__(self):
        self.archive = np.load("onlinefraud.npz", allow_pickle=True)
        self.fraud = self.archive["fraud"]
        self.columns = self.archive["column_names"]
        self.label = self.fraud[:, self.columns == "isFraud"][:, 0]
        self.features = self.fraud[:, [i for i, cname in enumerate(self.columns) if cname not in ["isFraud", "isFlaggedFraud", "nameDest", "nameOrig"]]]
        self.nsamples, self.nfeatures = self.features.shape
        self.is_training = np.random.uniform(0, 1, self.nsamples) < 0.8
    
    def generate(self, train:bool, size:int = 128):
        while True:
            indices = np.random.randint(0, self.nsamples, size*2)
            indices = indices[self.is_training[indices] == train]
            yield (
                (self.features[indices[:size]],),
                (self.label[indices[:size]],)
            )

dset = Dataset()

def gen_model():
    
    # input_layer = pipe = Input(8)
    # pipe = Dense(1)(pipe)
    # model = Model(inputs=[input_layer], outputs=[pipe])
    
    model = Sequential([
        Dense(units=1000, input_shape=[8]),
        layers.Activation('relu'),
        Dense(1),
        layers.Activation('sigmoid')
        
    ])
    
    
    model.compile("adam", "mse")
    return model

model = gen_model()
model.fit(
    dset.generate(True, size=128),
    steps_per_epoch=10000,
    epochs=10
)
code.interact(local=vars())
