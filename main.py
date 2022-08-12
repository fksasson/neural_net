import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
import neuralNet1
from timeit import timeit
import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2


def Get_data():
    nnfs.init()

    URL = 'http://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    if not os.path.isfile(FILE):
        print(f'Download {URL} and saving as {FILE}')
        urllib.request.urlretrieve(URL, FILE)

    print('Unzipping..')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

    print('Done!')



def load_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data(path):
    X, y = load_dataset('train', path)
    X_test, y_test = load_dataset('test', path)

    return X, y, X_test, y_test


def Main():
    print('Creating data')
    X, y, X_test, y_test = create_data('fashion_mnist_images')
    np.set_printoptions(linewidth=200)
    # print(X[0])

    # scale features min/max [-1,1]
    print('scaling features')
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    print(X.min(), X.max())
    print(X.shape)

    # Flatten dataset
    print('flattening')
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(f'X features: {X.shape[1]}\n'
          f'X_test features: {X_test.shape[1]}')

    # shuffle
    print('shuffling')
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # instantiate the model
    model = neuralNet1.Model()

    # add layers
    model.add(neuralNet1.Layer_Dense(X.shape[1], 128))
    model.add(neuralNet1.Activation_ReLU())
    model.add(neuralNet1.Layer_Dense(128, 128))
    model.add(neuralNet1.Activation_ReLU())
    model.add(neuralNet1.Layer_Dense(128,128))
    model.add(neuralNet1.Activation_ReLU())
    model.add(neuralNet1.Layer_Dense(128, 10))
    model.add(neuralNet1.Activation_Softmax())

    # set loss, optimizer and accuracy
    model.set(
        loss=neuralNet1.Loss_CategoricalCrossentropy(),
        optimizer=neuralNet1.Optimizer_Adam(decay=1e-4,learning_rate=0.005),
        accuracy=neuralNet1.Accuracy_Categorical()
    )

    # finalize
    model.finalize()

    # train
    model.train(X, y, validation_data=(X_test, y_test),
                epochs=15, batch_size=256, print_every=100)

    # save model
    model.save('fashion_mnist.model')


def load_param_model():
    print('Creating data')
    X, y, X_test, y_test = create_data('fashion_mnist_images')
    # np.set_printoptions(linewidth=200)
    # print(X[0])

    # scale features [-1,1]
    print('scaling features')
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    print(X.min(), X.max())
    print(X.shape)

    # Flatten dataset
    print('flattening')
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(f'X features: {X.shape[1]}\n'
          f'X_test features: {X_test.shape[1]}')

    # shuffle
    print('shuffling')
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # instantiate the model
    model = neuralNet1.Model()

    # add layers
    model.add(neuralNet1.Layer_Dense(X.shape[1], 128))
    model.add(neuralNet1.Activation_ReLU())
    model.add(neuralNet1.Layer_Dense(128, 128))
    model.add(neuralNet1.Activation_ReLU())
    model.add(neuralNet1.Layer_Dense(128, 10))
    model.add(neuralNet1.Activation_Softmax())

    # set loss, optimizer and accuracy
    model.set(
        loss=neuralNet1.Loss_CategoricalCrossentropy(),
        optimizer=neuralNet1.Optimizer_Adam(decay=1e-4),
        accuracy=neuralNet1.Accuracy_Categorical()
    )

    # finalize
    model.finalize()

    # load parameters
    model.load_parameters('fashion_mnist.params')

    # evaluate
    model.evaluate(X_test, y_test)


def load_model():
    # print('Creating data')
    # X, y, X_test, y_test = create_data('fashion_mnist_images')
    # # np.set_printoptions(linewidth=200)
    # # print(X[0])
    #
    # # scale features [-1,1]
    # print('scaling features')
    # X = (X.astype(np.float32) - 127.5) / 127.5
    # X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    # print(X.min(), X.max())
    # print(X.shape)
    #
    # # Flatten dataset
    # print('flattening')
    # X = X.reshape(X.shape[0], -1)
    # X_test = X_test.reshape(X_test.shape[0], -1)
    # print(f'X features: {X.shape[1]}\n'
    #       f'X_test features: {X_test.shape[1]}')
    #
    # # shuffle
    # print('shuffling')
    # keys = np.array(range(X.shape[0]))
    # np.random.shuffle(keys)
    # X = X[keys]
    # y = y[keys]

    model = neuralNet1.Model.load('fashion_mnist.model')

    shirt =cv2.imread("71hlZVUtDuL._AC_UX385_.jpg", cv2.IMREAD_GRAYSCALE)
    shirt = cv2.resize(shirt, (28,28))
    plt.imshow(shirt, cmap='gray')
    plt.show()
    shirt = (shirt.reshape(1,-1).astype(np.float32) - 127.5)/ 127.5




    confidences = model.predict(shirt)
    predictions = model.output_layer_activation.predictions(confidences)
    for prediction in predictions:
        print(fashion_mnist_labels[prediction])


fashion_mnist_labels = {
    0: 'T_shirts/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

if __name__ == "__main__":
    # Get_data()
    # opencv()
    # Main()
    load_model()
