# importing libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# initializing plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(2)
fig.tight_layout(pad=1.0)
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
A = []
F = []

# normalizing data
def normalize(df):
    for x in df.drop(columns = 'AboveMedianPrice'):
        df[x] = (df[x] - df[x].mean()) / df[x].std()
    return df

# importing the dataset
df = pd.read_csv('housepricedata.csv')
df = normalize(df)

#splitting the data into testing and training data
def split(df):
    test_df = df.sample(frac = 0.2)
    train_df = df.drop(test_df.index)

    x = train_df.drop(columns = 'AboveMedianPrice').values
    x = [np.reshape(i,(-1,1)) for i in x]
    y = train_df['AboveMedianPrice'].values
    training_data = list(zip(x,y))

    x = test_df.drop(columns = 'AboveMedianPrice').values
    x = [np.reshape(i,(-1,1)) for i in x]
    y = test_df['AboveMedianPrice'].values
    test_data = zip(x,y)

    return training_data, test_data

class Network(object):

    # initializing weights (He normal) and biases (zero)
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y,1)) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) * np.sqrt(2/x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # implementing feed forward
    def feedforward(self, a):
        n = self.num_layers
        for b, w in zip(self.biases, self.weights):
            n -= 1
            if n == 1:
                a = sigmoid(np.dot(w, a)+b)
            else:
                a = relu(np.dot(w,a)+b)
        return a

    # implementing mini batch gradient descent
    def SGD(self, training_data, epochs, batch_size, eta,
            test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k+batch_size]
                for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                t1, f1 = self.evaluate(test_data)
                print("Epoch {} : {} / {}".format(j,t1,n_test));
                test_accuracy.append(t1/n_test)
                e = self.loss(test_data)
                test_loss.append(e[0][0])
            else:
                print("Epoch {} complete".format(j))
            
            e = self.loss(training_data)
            train_loss.append(e[0][0])

            t2, f2 = self.evaluate(training_data)
            train_accuracy.append(t2/n)

            if j == epochs - 1:
                A.append(t1/n_test)
                F.append(f1)
                print('Accuracy =', '%.3f' % (t1/n_test), 'F-score =', '%.3f' % f1)
                plot_function()
                train_loss.clear()
                test_loss.clear()
                train_accuracy.clear()
                test_accuracy.clear()
               

    # updating mini batch
    def update_batch(self, batch, eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, delta_grad_w)]
        self.weights = [w-(eta/len(batch))*gw
                        for w, gw in zip(self.weights, grad_w)]
        self.biases = [b-(eta/len(batch))*gb
                       for b, gb in zip(self.biases, grad_b)]

    # implementing backpropagation 
    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x]
        zs = []
        n = self.num_layers
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            n -= 1
            if n == 1:
                activation = sigmoid(z)
            else:
                activation = relu(z)
            activations.append(activation)

        # backward pass
        delta = self.loss_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            rp = relu_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * rp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].T)
        return (grad_b, grad_w)

    # calculating accuracy and fscore
    def evaluate(self, data):     
        X = np.array([np.round(self.feedforward(x)) for (x,y) in data]).reshape(len(data),1)
        Y = np.array([y for (x,y) in data]).reshape(len(data),1)

        correct = ((X==Y) * 1).sum()

        precision = (X * Y).sum()/X.sum()
        recall = (X * Y).sum()/Y.sum()
        fscore = (2 * precision * recall)/(precision + recall)

        return correct, fscore

    # defining loss and loss derivative
    def loss(self, data):
        e = 0
        for x,y in data:
            e += 0.5 * (self.feedforward(x) - y) ** 2
        e /= len(data)
        return e

    def loss_derivative(self, output_activations, y):
        return (output_activations-y)

# defining sigmoid, relu and their derivatives
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return z * (z > 0)

def relu_prime(z):
    return 1 * (z > 0)

# plotting loss and accuracy graphs
def plot_function():
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Number of epochs')
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Number of epochs')
    ax1.plot(train_loss, label = 'train')
    ax1.plot(test_loss, label = 'test')
    ax1.legend()
    ax2.plot(train_accuracy, label = 'train')
    ax2.plot(test_accuracy, label = 'test')
    ax2.legend()
    plt.draw()
    plt.pause(1)
    ax1.clear()
    ax2.clear()

# running neural network multiple times
for i in range(10):
    training_data, test_data = split(df)
    net = Network([10,5,5,1])
    net.SGD(training_data, 100, 10, 0.01, test_data=test_data)
    print()

# Accuracy
A_mean = np.mean(A)
A_std = np.std(A)

print ('Accuracy = ', '%.3f' % A_mean, '±', '%.3f' % A_std)

# F-score
F_mean = np.mean(F)
F_std = np.std(F)

print ('F-score = ', '%.3f' % F_mean, '±', '%.3f' % F_std)