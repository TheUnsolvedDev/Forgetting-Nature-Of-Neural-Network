import tensorflow as tf
import numpy as np
from tqdm import tqdm

metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
    tf.keras.metrics.SparseCategoricalCrossentropy('loss')
]


def lenet5():
    # input = tf.keras.Input(shape=(28, 28, 1))
    # conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5,
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(input)
    # maxpool2 = tf.keras.layers.MaxPool2D(
    #     pool_size=(2, 2), strides=None, padding='valid')(conv1)
    # droupout1 = tf.keras.layers.Dropout(0.25)(maxpool2)

    # conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=5,
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='valid')(maxpool2)
    # maxpool3 = tf.keras.layers.MaxPool2D(
    #     pool_size=(2, 2), strides=None, padding='valid')(conv3)
    # dropout2 = tf.keras.layers.Dropout(0.25)(maxpool3)

    # flat = tf.keras.layers.Flatten()(dropout2)
    # fc1 = tf.keras.layers.Dense(units=240, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(flat)
    # fc2 = tf.keras.layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(fc1)
    # final = tf.keras.layers.Dense(units=10)(fc2)

    # model = tf.keras.Model(inputs=input, outputs=final)

    # return model

    inputs = tf.keras.layers.Input(shape=(28, 28))
    flat = tf.keras.layers.Flatten()(inputs)
    flat = tf.keras.layers.Dense(240, activation='tanh')(flat)
    output = tf.keras.layers.Dense(10, activation='linear')(flat)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def ewc_fisher_matrix(datas, labels, model, samples=400):
    fisher = [tf.zeros_like(tensor) for tensor in model.trainable_weights]
    length = len(datas)

    for data in datas:
        for sample in tqdm(range(samples)):
            num = np.random.randint(data.shape[0])
            with tf.GradientTape() as tape:
                probs = (model(tf.expand_dims(data[num], axis=0)))
                log_likelyhood = tf.nn.log_softmax(probs)

            derv = tape.gradient(log_likelyhood, model.weights)
            fisher = [(fis + dv**2) for fis, dv in zip(fisher, derv)]

        fisher = [fish/(samples) for fish in fisher]
    return fisher


if __name__ == '__main__':
    model = lenet5()
    model.summary()
