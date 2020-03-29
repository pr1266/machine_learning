
#! now we want implement logistic regression class
#! just using numpy package
#! created by pourya pooryeganeh
#! 2018

import numpy as np

class LogisticRegression:

    def __init__(self, X, Y):
        
        x = np.array(X)
        m, n = x.shape

        #! normalize data:
        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)

        #! add a const column to x:
        const = np.array([1] * m).reshape(m, 1)
        self.X = np.append(const, x, axis = 1)

        self.Y = np.array(Y)

        self.num_iter = 1500
        self.learning_rate = 0.005
        self.landa = 0.1
        self.theta = np.array([0, 0] * (n + 1))

        self.treshold = 0.5

    def sigmoid(self, z):

        g = 1 / (1 + np.exp(-z))
        return g

    def gradiant_descent(self):

        m, n = self.X.shape
        for i in range(self.num_iter):
            theta_temp = self.theta
            h_theta = self.sigmoid(np.dot(self.X, self.theta))
            diff = h_theta - self.Y

            self.theta[0] = theta_temp[0] - self.learning_rate * sum(diff * self.X[:, 0])

            for j in range(1, n):
                val = theta_tmep[j] - self.alpha * (1.0/m) * sum(diff * self.X[:, 0])
                self.theta[j] = val
            
    def predict(self, X):

        x = np.array(X)
        m, n = x.shape

        x = (x - self.mean) / self.std

        cost = np.array([1] * m).reshape(m, 1)

        pred = self.sigmoid(np.dot(X, self.theta))

        if pred < self.treshold:
            return 0
        return 1