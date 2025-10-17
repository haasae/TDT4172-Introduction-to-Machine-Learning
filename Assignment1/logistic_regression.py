import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class LogisticRegression:
    
    def __init__(self, learning_rate=0.001, epochs=100):

        self.learning_rate = learning_rate 
        self.epochs = epochs # amount of iterations
        self.weights, self.bias = None, None

        self.losses = [] # mse per epoch log
        self.accuracy = [] # mean residual log

    # Returns gradient for weight (w) and bias (b) for loss-function
    def calc_gradients(self, x, y, y_pred):
        gradient_w = -np.dot(x.T, (y - y_pred)) / len(y)  
        gradient_b = -np.mean(y - y_pred)  
        return gradient_w, gradient_b
    
    # Updates weight and bias based on the gradients
    def update_parameters(self, grad_w, grad_b):
        self.weights-= self.learning_rate*grad_w
        self.bias-= self.learning_rate*grad_b

    # Uses loss function 1/n*sum(y-y_hat)^2
    def calc_loss(self, y, y_pred):
        return np.mean(-y*np.log(y_pred) - (1 - y)*np.log(1 - y_pred))

    # Average residual
    def mean_residual(self, y, y_pred):
        y_hat = (y_pred >= 0.5).astype(int)
        return np.mean(y_hat == y) # Returns the accuracy
    
    def sigmoid_function(self, z):
        return 1.0 /(1.0 + np.exp(-z))

    # Main calculations
    def fit(self, X, y):
        # Init weight and bias
        if self.weights is None:
            self.weights = np.zeros(X.shape[1]) # initialized to zero-vector with length = amount of features
        if self.bias is None:
            self.bias = 0.0

        # Gradient descent-loop (Training-loop)
        for _ in range(self.epochs):
            y_pred = self.sigmoid_function(np.dot(X, self.weights) + self.bias)
            gradient_w, gradient_b = self.calc_gradients(X, y, y_pred) 
            self.update_parameters(gradient_w, gradient_b)
            loss = self.calc_loss(y, y_pred)
            accuracy = self.mean_residual(y, y_pred)
            self.losses.append(loss)
            self.accuracy.append(accuracy)

    # Returns matrice-vector-product + bias
    def predict(self, X): 
        probs = self.sigmoid_function(np.dot(X, self.weights) + self.bias)
        return (probs >= 0.5).astype(int)
        
    def plot_ROC_curve(self, X, y, show=True, ax=None, label=None):
        y_scores_lin = np.dot(X, self.weights) + self.bias # The linear score (for each sample x_i, compute z_i = x_i*w + b
        y_scores_prob = self.sigmoid_function(y_scores_lin) # Converts to probabilities
        fpr, tpr, _ = metrics.roc_curve(y, y_scores_prob)
        auc = metrics.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")   # random classifier baseline (diagonal)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
    
        plt.legend(loc="lower right"); plt.show()
        