"""
Michael Samon
Created on 11-27-18
Description: Implementation of ML model to solve suduko puzzles in Keras
"""

from data_prep import DataReader
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras.backend as K
import keras
import tensorflow as tf
from datetime import datetime
import numpy as np


# full path to repo
BASE_PATH = "/home/michael_samon/sudoku-solver"


# TODO:
# custom loss function implementation... more info in README file
# expects two arrays shape (81,) as input
def puzzle_loss(y_true, y_pred):

    possible = 1/81
    wrong_preds = K.variable(0)
    for i in range(81):
        if y_true[i] != y_pred[i]:
            wrong_preds += 1

    # return wrong preds as a percent out of 81 possible squares
    return wrong_preds * possible


adam = keras.optimizers.adam(lr=0.0001)


# model creation and training
def build_model(x_train, y_train, op):

    model = Sequential()
    model.add(Dense(4000, activation="relu", input_dim=810))
    model.add(Dense(4000, activation="relu"))
    model.add(Dense(4000, activation="relu"))
    model.add(Dense(4000, activation="relu"))
    #model.add(Dense(3000, activation="tanh"))
   # model.add(Dense(3000, activation="tanh"))
    model.add(Dense(810, activation="sigmoid"))

    model.compile(loss="categorical_crossentropy", optimizer=op)

    callbacks = [#keras.callbacks.TensorBoard(
                  #              log_dir="./logs/"+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
                 keras.callbacks.ReduceLROnPlateau(monitor="loss"),
                 # keras.callbacks.EarlyStopping(monitor="loss", patience=10),
                 keras.callbacks.ModelCheckpoint("model_" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".h5",
                                                 monitor='val_loss', save_best_only=True)
                 ]

    with tf.device('/gpu:0'):
        model.fit(x_train,
                y_train,
                epochs=20,
                batch_size=50,
                validation_split=0.1,
                callbacks=callbacks)

    return model


# gather all data
reader = DataReader(BASE_PATH)
x, y = reader.get_data()

# reserve first 100000 for testing
x_train = to_categorical(x[100000:150000]).reshape((50000, 810))
y_train = to_categorical(y[100000:150000]).reshape((50000, 810))

x_test = to_categorical(x[:100000]).reshape((100000, 810))
y_test = y[:100000]

model = build_model(x_train, y_train, adam)
model.save("model-v2_" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".h5")

#model = keras.models.load_model("model_2018-11-29_04_01_38.h5")
preds = model.predict(x_test[:3]).reshape((3, 81, 10))

preds = np.argmax(preds, axis=2).reshape((100000, 81, 10))
correct = 0
for i, pred in enumerate(preds):
    compared = (pred == y_test[i])
    if False not in compared:
        correct += 1

print("\nModel correctly solved {}/100000 puzzles".format(correct))
