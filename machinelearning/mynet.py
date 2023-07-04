import numpy as np
import random

class NeuralNetwork:

    def __init__(self, sizes, weights=None, biases=None):
        self.sizes = sizes # array of sizes of layers
        self.num_layers = len(sizes)
        if weights == None:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # each entry is weights from one layer to next, num_layers-1 entries
        else:
            self.weights = weights
        if biases == None:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            self.biases = biases

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    # loss and cost may be wrong
    def loss(self, a, y):
        """
        Returns loss given a: ouptut of neural net, and y: expected output
        This works on a single training example
        """
        return 1/2 * np.linalg.norm(y - a) ** 2

    def cost_derivative(self, a, y):
        """
        Returns costs derivative given a: output of neural net, and y: expected output
        """
        return a - y
    
    def feedforward(self, x):
        """
        Takes a vector x of size nx1 (a column vector) where n is number of input neurons
        Returns output layer
        """
        a = x
        for i in range(self.num_layers-1):
            z = np.matmul(self.weights[i], a) + self.biases[i]
            a = self.sigmoid(z)
        return a

    def train(self, training_data, epochs, eta, mini_batch_size=32, test_data=None):
        """
        Trains the neural network using the supplied training_data.
        An epoch is when the entire training data is covered
        Each epoch is split up into training by mini batches, where weights are updated after each mini batch
        using grad_descent and backprpop.
        Status updates may also be used by supplying test_data
        """
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i: i+mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.grad_descent(mini_batch, eta)
            if test_data:
                print(f"Epoch {epoch} complete: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

    def grad_descent(self, mini_batch, eta):
        """
        Updates the weights and biases of the network using gradients calculated by the backprop method
        """
        n_w = [np.zeros(w.shape) for w in self.weights]
        n_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            d_w, d_b = self.backprop(x, y)
            n_w = [nw + dw for nw, dw in zip(n_w, d_w)]
            n_b = [nb + db for nb, db in zip(n_b, d_b)]
        self.weights = [w - (eta / len(mini_batch)) * d_w for w, d_w in zip(self.weights, n_w)]
        self.biases = [w - (eta / len(mini_batch)) * d_b for w, d_b in zip(self.biases, n_b)]

    def backprop(self, x, y):
        """
        Calculates gradients (derivatives of weights/biases with respect to cost function) using
        backpropogation
        """
        # list of gradients of weights: each entry is matrix of derivatives of weights from one layer to next
        d_w = [np.zeros(w.shape) for w in self.weights]
        # list of gradients of biases
        d_b = [np.zeros(b.shape) for b in self.biases]
        # list of activations
        a = [x]
        # list of outputs before activation function
        zs = []
        for i in range(self.num_layers-1):
            z = np.matmul(self.weights[i], a[i]) + self.biases[i]
            zs.append(z)
            a.append(self.sigmoid(z))
        # delta is the derivative of the cost function with respect to the zs in the last layer
        delta = np.multiply(self.cost_derivative(a[-1], y), self.sigmoid_prime(zs[-1]))
        d_w[-1] = np.matmul(delta, a[-2].transpose())
        d_b[-1] = delta
        for i in range(2, self.num_layers):
            # calculate delta of previous layer given delta currently known
            delta = np.matmul(self.weights[-i+1].transpose(), delta) * self.sigmoid_prime(zs[-i])
            # calculate gradients using this delta
            d_w[-i] = np.matmul(delta, a[-i-1].transpose())
            d_b[-i] = delta
        return (d_w, d_b)

    def evaluate(self, test_data):
        """
        Given test data, returns how many items are correctly predicted
        """
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in results)