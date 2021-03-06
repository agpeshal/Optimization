# -*- coding: utf-8 -*-
"""

Gradient descent
"""

import numpy as np

def calculate_mse(e):
    """Calculate the mean squared error for vector e."""
    
    return 0.5*np.mean(e**2)

def compute_gradient(b, A, x):
    """Compute the gradient."""
    n = 1.0/len(b)
    grad = n*np.matmul(A.T, np.dot(A,x)- b)
    err = np.dot(A,x)- b
    return grad, err

def gradient_descent(b, A, initial_x, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    objectives = []
    x = initial_x
    for n_iter in range(max_iters):
        
        grad, err = compute_gradient(b, A, x)
        obj = calculate_mse(err)
        
        x = x - gamma*grad
        # store x and objective function value
        xs.append(x)
        objectives.append(obj)
        print("Gradient Descent({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))

    return objectives, xs