from operator import iadd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from dataset import *
from model import *

tf.keras.backend.set_floatx('float32')


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, name1, name2) -> None:
        super(CustomCallback).__init__()
        self.task_1_accuracy = []
        self.task_2_accuracy = []
        self.name1 = name1
        self.name2 = name2

    def on_epoch_end(self, epoch, logs=None):
        self.task_1_accuracy.append(logs['accuracy'])
        self.task_2_accuracy.append(logs['val_accuracy'])

    def on_train_end(self, logs=None):
        plt.plot(self.task_1_accuracy, label=self.name1)
        plt.plot(self.task_2_accuracy, label=self.name2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelA*.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
    CustomCallback('A','B')
]

ewc_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelB.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
    CustomCallback('B', 'A')
]


def plot_result(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def train(train_data, train_labels, validation_data, validation_labels, epochs=15, model=None):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer='Adam', metrics=metrics)
    history = model.fit(train_data, train_labels, validation_data=(
        validation_data, validation_labels), epochs=epochs, callbacks=callbacks)
    # plot_result(history, 'loss')
    # plot_result(history, 'accuracy')
    return model


@ tf.function
def ewc_loss_fn(y_true, y_pred, lam=10):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_loss = loss(y_true, y_pred)
    for i in range(len(theta)):
        diff = tf.reduce_sum(I[i]*tf.square(theta[i] - theta_star[i]))
        total_loss += (lam/2)*diff
    return total_loss


def train_ewc(train_data, train_labels, validation_data, validation_labels, batch_size=500, epochs=15, callbacks=None, model=None):
    model.compile(loss=ewc_loss_fn, optimizer='Adam', metrics=metrics)
    history = model.fit(train_data, train_labels, validation_data=(
        validation_data, validation_labels), epochs=epochs, callbacks=callbacks)
    return model


if __name__ == '__main__':
    obj = Dataset()
    A1, A2 = obj.task_A()
    B1, B2 = obj.task_B()
    C1, C2 = obj.task_C()

    star = lenet5()
    star = train(A1, A2, B1, B2, model=star)

    theta, theta_star = star.weights, star.get_weights()
    I = ewc_fisher_matrix([A1], [A2], star)

    star = train_ewc(B1, B2, A1, A2, model=star,callbacks = ewc_callbacks)

    theta, theta_star = star.weights, star.get_weights()
    I = ewc_fisher_matrix([A1,B1], [A2,B2], star)

    star = train_ewc(C1, C2, B1, B2, model=star,callbacks = ewc_callbacks)

    star.evaluate(A1,A2)
    star.evaluate(B1,B2)
    star.evaluate(C1,C2)
