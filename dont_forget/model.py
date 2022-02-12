import tensorflow as tf
import numpy as np

metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
    tf.keras.metrics.SparseCategoricalCrossentropy('loss')
]


def lenet5():
    input = tf.keras.Input(shape=(28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                   activation='relu', padding='same')(input)
    maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid')(conv1)
    droupout1 = tf.keras.layers.Dropout(0.25)(maxpool2)

    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                                   activation='relu', padding='valid')(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid')(conv3)
    dropout2 = tf.keras.layers.Dropout(0.25)(maxpool3)

    flat = tf.keras.layers.Flatten()(dropout2)
    fc1 = tf.keras.layers.Dense(units=240, activation='relu')(flat)
    fc2 = tf.keras.layers.Dense(units=128, activation='relu')(fc1)
    final = tf.keras.layers.Dense(units=10)(fc2)

    model = tf.keras.Model(inputs=input, outputs=final)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer='Adam', metrics=metrics)

    return model


def compute_elastic_penalty(F, theta, theta_A, alpha=25):
    penalty = 0
    for i, theta_i in enumerate(theta):
        penalty_this = tf.math.reduce_sum(F[i] * (theta_i - theta_A[i]) ** 2)
        penalty += penalty_this
    # print(penalty)
    return 0.5*alpha*penalty


if __name__ == '__main__':
    model = lenet5()
    model.summary()
