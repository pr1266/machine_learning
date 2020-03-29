
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
        slef.theta = np.array([0, 0] * (n + 1))