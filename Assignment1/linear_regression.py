import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, epochs=100):

        self.learning_rate = learning_rate 
        self.epochs = epochs # amount of iterations
        self.weights, self.bias = None, None

        self.losses = [] # mse per epoch log
        self.accuracy = [] # mean residual log

    # Returns gradient for weight (w) and bias (b) for loss-function
    def calc_gradients(self, x, y, y_pred):
        gradient_w = (2/len(y)) * np.dot(x.T, (y_pred-y))  # dL/dw
        gradient_b = (2/len(y)) * np.sum(y_pred-y)  # dL/db
        return gradient_w, gradient_b
    
    # Updates weight and bias based on the gradients
    def update_parameters(self, grad_w, grad_b):
        self.weights-= self.learning_rate*grad_w
        self.bias-= self.learning_rate*grad_b

    # Uses loss function 1/n*sum(y-y_hat)^2
    def calc_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    # Average residual
    def mean_residual(self, y, y_pred):
        return np.mean(y - y_pred)

    # Main calculations
    def fit(self, X, y):
        # Init weight and bias
        if self.weights is None:
            self.weights = np.zeros(X.shape[1]) # initialized to zero-vector with length = amount of features
        if self.bias is None:
            self.bias = 0.0

        # Gradient descent-loop (Training-loop)
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            gradient_w, gradient_b = self.calc_gradients(X, y, y_pred) 
            self.update_parameters(gradient_w, gradient_b)
            loss = self.calc_loss(y, y_pred)
            accuracy = self.mean_residual(y, y_pred)
            self.losses.append(loss)
            self.accuracy.append(accuracy)

    # Returns matrice-vector-product + bias
    def predict(self, X): 
        prediction= np.dot(X, self.weights) + self.bias
        return prediction




