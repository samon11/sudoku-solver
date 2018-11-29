"""
Michael Samon
Created on 11-27-18
Description: Implementation of ML model to solve suduko puzzles in Keras
"""

from data_prep import DataReader
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import keras
from datetime import datetime


# full path to repo
BASE_PATH = "/home/user/keras/sudoku-solver"


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


# model creation and training
def build_model(x_train, y_train):

    model = Sequential()
    model.add(Dense(500, activation="sigmoid", input_dim=81))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(500, activation="tanh"))
    model.add(Dense(500, activation="sigmoid"))
    model.add(Dense(81))

    model.compile(loss="mae", optimizer="adam")

    callbacks = [keras.callbacks.TensorBoard(
                                log_dir="./logs/"+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
                 #keras.callbacks.EarlyStopping(monitor="loss", patience=10),
                 keras.callbacks.ModelCheckpoint("model_"+ datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".h5", monitor='val_loss', save_best_only=True)
                 ]

    model.fit(x_train,
              y_train,
              epochs=9000,
              batch_size=750,
              validation_split=0.1,
              callbacks=callbacks)

    return model


# gather all data
reader = DataReader(BASE_PATH)
x, y = reader.get_data()

# reserve first 100000 for testing
x_train = x[100000:]
y_train = y[100000:]

x_test = x[:100000]
y_test = y[:100000]

model = build_model(x_train, y_train)
model.save("model_"+ datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".h5")
preds = model.predict(x_test)

correct = 0
for i, pred in enumerate(preds):
    compared = (pred == y_test[i])
    if False not in compared:
        correct += 1

print("\nModel correctly solved {}/100000 puzzles".format(correct))
