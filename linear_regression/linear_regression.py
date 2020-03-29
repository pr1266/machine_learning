
#! now we are goin to implement a simple linear regression
#! with gradiant descent optimization algorithm from scratch
#TODO created by pourya pooryeganeh


#? import packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#! pre processing data considering data import 
def data_prepare():
    
    x = 2 * np.random.rand(100,1)
    y = 4 + 3 * x + np.random.randn(100,1)

    return x, y

#! now we implement the gradiant descent algorithm:
def gradiant_descent(x, y, learning_rate, num_iteration):

    J = []
    n = len(x)
    
    m_curr = b_curr = 0

    for _ in range(num_iteration):

        h = m_curr * x + b_curr
        cost = (1/n) * sum(val**2 for val in (y - h))
        J.append(cost)

        dm = (-2/n) * sum(x * (y - h))
        db = (-2/n) * sum(y - h)

        m_curr -= dm * learning_rate
        b_curr -= db * learning_rate
        
    return m_curr, b_curr, J

#def predict()

if __name__ == '__main__':

    #TODO first prepare data: x , y
    x, y = data_prepare()
    
    #? now ploting raw data:
    # plt.scatter(x, y)
    # plt.show()
    
    #! get the first and second coeff and also J function history
    m, b, J = gradiant_descent(x, y, 0.005, 1500)
    
    
    #TODO ploting cost history:
    plt.plot(J)
    plt.xlabel("iterations")
    plt.ylabel("MSE(min_squared_error)")
    plt.show()

    y_pred = m * x + b
    print('M : ', m)
    print('B : ', b)
    print('Last Cost : ', J[-1])
    #! plotting model
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color = 'r')
    plt.show()

    
    


