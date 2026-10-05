import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/train.csv')

data = np.array(data)
np.random.shuffle(data)
m, n=  data.shape # m: data sample, n: features

# Spit data
train_data = data[0: int(0.8 * m), :]
val_data = data[int(0.8 * m): m, :]

# I want this in the format where every column corresponds to one image
X_train = train_data[:, 1:].T
X_train = X_train / 255.0 # Scale
Y_train = train_data[:, 0]

X_val = val_data[:, 1:].T
X_val = X_val / 255.0
y_val = val_data[:, 0]

def initialize_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def ReLU(X):
    return np.maximum(X, 0)

def softmax_calculator(Z):
    Z = Z - np.max(Z, axis = 0, keepdims = True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis = 0, keepdims = True)

def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax_calculator(Z2)
    return Z1, A1, Z2, A2

def one_hot_converter(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot_converter(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, interations, W1 = None, B1 = None, W2 = None, B2 = None):
    if W1 is None:
        W1, B1, W2, B2 = initialize_parameters()
    for i in range(interations):
        Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

        if ((i + 1) % 20 == 0):
            print('Iteration number: ', i + 1)
            print('Accuracy = ', get_accuracy(get_predictions(A2), Y))
    return W1, B1, W2, B2

if __name__ == '__main__':
    # If you want train from scratch with random weights you can remove this line
    model = np.load('mnist_model.npz')
    W1 = model['W1']
    B1 = model['B1']
    W2 = model['W2']
    B2 = model['B2']
    # Train
    W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.01, 1500, W1, B1, W2, B2)
    np.savez(
        "mnist_model.npz",
        W1=W1,
        B1=B1,
        W2=W2,
        B2=B2
    )