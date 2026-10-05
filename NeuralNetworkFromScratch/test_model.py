import numpy as np
import NNFS 
import pandas as pd
import matplotlib.pyplot as plt

model = np.load('mnist_model.npz')
W1 = model['W1']
B1 = model['B1']
W2 = model['W2']
B2 = model['B2']

data = pd.read_csv('data/test.csv')
data = np.array(data).T

index = 235
X_test = data[1:, :]
y_test = data[0, :]

_, _, _, A2test = NNFS.forward_propagation(W1, B1, W2, B2, X_test)

test_predictions = NNFS.get_predictions(A2test)

print('Test Accuracy: ', NNFS.get_accuracy(test_predictions, y_test))