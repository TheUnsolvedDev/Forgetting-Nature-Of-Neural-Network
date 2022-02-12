from operator import iadd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from dataset import *
from model import *

tf.keras.backend.set_floatx('float32')

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelA*.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

ewc_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelB.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
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


def train(train_data, train_labels, validation_data, validation_labels, epochs=15):
    model = lenet5()
    history = model.fit(train_data, train_labels, validation_data=(
        validation_data, validation_labels), epochs=epochs, callbacks=callbacks)
    plot_result(history, 'loss')
    plot_result(history, 'accuracy')
    return history


def test(weights, data, labels):
    model = lenet5()
    model.load_weights(weights)

    score = model.evaluate(data, labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def fisher_matrix(model, images, labels):
    inputs = images
    y = labels
    weights = model.trainable_weights
    variance = [tf.zeros_like(tensor) for tensor in weights]
    batch_size = 50
    parts = 0.01

    indices = int(parts*np.floor(len(images)/batch_size))
    for i in tqdm(range(indices)):
        data, label = images[i*batch_size:(
            i+1)*batch_size], labels[i*batch_size: (i+1)*batch_size]
        with tf.GradientTape() as tape:
            log_likelihood = (tf.nn.log_softmax(model(data)))

        gradients = tape.gradient(log_likelihood, weights)
        variance = [var + tf.reduce_mean(grad ** 2, axis=0)/batch_size
                    for var, grad in zip(variance, gradients)]

    fisher_diagonal = variance
    return fisher_diagonal


def train_ewc(I, train_data, train_labels, validation_data, validation_labels, batch_size=500, epochs=100):
    accuracy = metrics[0]
    loss = metrics[1]
    model = lenet5()
    star_model = lenet5()
    star_model.load_weights('modelA*.h5')

    for epoch in range(epochs):
        accuracy.reset_states()
        loss.reset_states()

        indices = int(np.floor(len(train_data)/batch_size))
        for i in range(indices):
            data, labels = train_data[i*batch_size:(
                i+1)*batch_size], train_labels[i*batch_size: (i+1)*batch_size]

            with tf.GradientTape() as tape:
                pred = model(data)
                total_loss = loss(labels, pred) + compute_elastic_penalty(I,
                                                                          model.trainable_variables, star_model.trainable_variables)
            # print(total_loss,1000)
            grads = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            accuracy.update_state(labels, pred)
            loss.update_state(labels, pred)
            print("\rEpoch: {}, Batch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(
                epoch+1, i+1, loss.result().numpy(), accuracy.result().numpy()), flush=True, end='')

        print("")
        accuracy.reset_states()
        loss.reset_states()

        indices = int(np.floor(len(train_data)/batch_size))
        for i in range(indices):
            val_data, val_labels = validation_data[i*batch_size:(
                i+1)*batch_size], validation_labels[i*batch_size: (i+1)*batch_size]
            pred = model(val_data)

            accuracy.update_state(labels, pred)
            loss.update_state(labels, pred)
            print("\rEpoch: {}, Batch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(
                epoch+1, i+1, loss.result().numpy(), accuracy.result().numpy()), flush=True, end='')
        print("")
        
    model.evaluate(train_data,train_labels)
    model.evaluate(validation_data,validation_labels)
    model.save_weights('modelB.h5')


if __name__ == '__main__':
    obj = Dataset()
    A1, A2, A3, A4 = obj.task_A()
    train(A1, A2, A3, A4)
    test('modelA*.h5', A3, A4)

    model = lenet5()
    model.load_weights('modelA*.h5')
    I = (fisher_matrix(model, A1, A2))
    print(I)

    B1, B2, B3, B4 = obj.task_B()
    train_ewc(I, B1, B2, A1, A2)
    test('modelB.h5',  A3, A4)
