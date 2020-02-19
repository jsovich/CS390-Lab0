import os
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Sigmoid function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid prime function.
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    # ReLU function
    def __relu(self, x):
        return np.maximum(0, x)

    # ReLU prime function
    def __reluDerviative(self, x):
        if x > 0:
            return 1
        else:
            return 0

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=2, minibatches=False, mbs=100):
        xVals = xVals.reshape(xVals.shape[0], IMAGE_SIZE)
        if minibatches:
            for i in range(epochs):
                for j in self.__batchGenerator(xVals, mbs):
                    pass
            pass
        else:
            for i in range(epochs):
                print("Epoch :", i)
                for j in range(60000):
                    layer1, layer2 = self.__forward(xVals[j])
                    l2e = yVals[j] - layer2
                    l2d = l2e * self.__sigmoidDerivative(layer2)
                    l1e = l2d.dot(self.W2.T)
                    l1d = l1e * self.__sigmoidDerivative(layer1)
                    l1a = xVals[j].reshape(xVals[j].shape[0], 1).dot((l1d.reshape(l1d.shape[0], 1)).T) * self.lr
                    l2a = layer1.reshape(layer1.shape[0], 1).dot(l2d.reshape(l2d.shape[0], 1).T) * self.lr
                    self.W1 += l1a
                    self.W2 += l2a

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        xVals = xVals.reshape(xVals.shape[0], IMAGE_SIZE)
        _, layer2 = self.__forward(xVals)
        for pred in layer2:
            index = tf.argmax(pred)
            for i in range(len(pred)):
                if i != index:
                    pred[i] = 0
                else:
                    pred[i] = 1
        return layer2


class BuildTfNet():
    def __init__(self):
        self.model = keras.Sequential(keras.layers.Flatten())
        self.lossType = keras.losses.mean_squared_error
        self.inShape = (IMAGE_SIZE,)
        self.model.add(keras.layers.Dense(512, activation=tf.nn.sigmoid))
        self.model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))
        self.model.compile(optimizer='adam', loss=self.lossType, metrics=['accuracy'])

    def train(self, x, y, epochs):
        self.model.fit(x, y, epochs=epochs)

    def runTFModel(self, x):
        preds = self.model.predict(x)
        for pred in preds:
            index = tf.argmax(pred)
            for i in range(len(pred)):
                if i != index:
                    pred[i] = 0
                else:
                    pred[i] = 1
        return preds

    def eval(self, x, y):
        self.model.evaluate(x, y)


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    # range reduction
    xTrain = xTrain / 255.0
    # print(xTrain[0])
    xTest = xTest / 255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        neuralNet = NeuralNetwork_2Layer(IMAGE_SIZE, 10, 512)
        neuralNet.train(xTrain, yTrain)
        return neuralNet
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        tfNet = BuildTfNet()
        tfNet.train(xTrain, yTrain, 20)
        return tfNet
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return model.runTFModel(data)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
