import numpy as np
from sklearn.datasets import load_boston

# Load the data
data = load_boston()
X_ = data['data']
y = data['target']
y = y.reshape(y.shape[0], 1)

# data standardization
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
# Weight of input layer to hidden layer (n_features, n_hidden==> 10)
w1 = np.random.randn(n_features, n_hidden)
b1 = np.zeros(n_hidden)
# Weight of hidden layer to output layer (n_hidden==> 10, 1)
w2 = np.random.randn(n_hidden, 1)
b2 = np.zeros(1)


def Relu(x):
    return np.where(x < 0, 0, x)


learning_rate = 1e-6


# Loss function
def MSE_loss(y, y_hat):
    result = np.mean(np.square(y_hat - y))
    return result


# Linear Regression
def Linear(X, W1, b1):
    y = X.dot(W1) + b1
    return y


# iterations
epoch = 5000

for t in range(epoch):
    # forward propagation
    linear = Linear(X_, w1, b1)
    relu = Relu(linear)
    y_pred = Linear(relu, w2, b2)

    # compute loss
    loss = MSE_loss(y_pred, y)

    # backpropagation
    grad_y_pred = 2.0 * (y_pred)
    grad_w2 = relu.T.dot(grad_y_pred)
    grad_temp_relu = grad_y_pred.dot(w2.T)
    grad_temp_relu[linear < 0] = 0
    grad_w1 = X_.T.dot(grad_temp_relu)

    # update weights
    w1 -= grad_w1 * learning_rate
    w2 -= grad_w2 * learning_rate

# get the final weights
print('w1={} \n w2={}'.format(w1, w2))
