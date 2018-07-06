from __future__ import division
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import math, random
from functools import partial

# -*- coding: cp1252 -*- 
#***************************************************************************************************
# **********************Chapter 8. Gradient Descent*******************************************
#***************************************************************************************************

# NOTE:
# If a function has a unique global minimum, this procedure is likely to find it.
# If a function has multiple (local) minima, this procedure might 'find' the wrong one of them,
# in which case you might re-run the procedure from a variety of starting points.
# If a function has no minimum, then it's possible the procedure might go on forever.


# Estimating the Gradient:
# If f is a function of one variable, its derivative at a point x measures how f(x) changes when 
# we make a very small change to x.
# It is defined as the limit of the difference quotients:

def difference_quotient(f,x,h):
    return (f(x+h)-f(x))/h     # as h approaches zero.

def square(x):
    return x*x

# has the derivative. This is other way than doing difference_quotient for a very small h.
def derivative(x):
    return 2*x

derivative_estimate=partial(difference_quotient,square,h=.000001)
'''
x=range(-10,10)
y1=map(derivative,x)
y2=map(derivative_estimate,x)
print 'y1 is ', y1
print 'y2 is ', y2
'''

# When f is a function of many variables, it has multiple partial derivatives, each indicating how f changes
# when we make small changes in just one of the input variables. We calculate its ith partial derivative by treating 
# it as a function of just its ith variable, holding the other variables fixed:

def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) # add h to just the ith element of v
            for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

# after which we can estimate the gradient the same way:
def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
                for i, _ in enumerate(v)]


# NOTE:
# A major drawback to this "estimate using difference quotients" approach is that it's computationally expensive.
# If v has length n, estimate_gradient has to evaluate f on 2n different inputs.
# If you're repeatedly estimating gradients. you're doing a whole lot of extra work.


# Using the Gradient:
# It's easy to see that the sum_of_squares function is smallest when its input v is a vector of zeroes.
# But imagine we didn't know that. Let's use gradients to find the minimum among all three-dimensional vectors.
# We'll just pick a random starting point and then take tiny steps in the opposite direction of the gradient
# until we reach a point where the gradient is very small:

def step(v, gradient, step_size):
    """move step_size in the direction from v"""
    return[v_i+step_size*gradient_i for v_i,gradient_i in zip(v,gradient)]

def sum_of_squares_gradient(v):
    """derivative of sum of squares function at a point"""
    return [2*v_i for v_i in v]
'''
# pick a random starting point
v = [random.randint(-10,10) for i in range(3)]
tolerance = 0.0000001
print 'start v is ', v
'''

'''
while True:
    global v
    gradient = sum_of_squares_gradient(v)
    next_v = step(v,gradient,-0.001)           
    if distance(next_v,v) < tolerance:
        break
    v=next_v
# print 'minima is at', v
'''


# It is possible that certain step sizes will result in invalid inputs for our function. So we'll need to create
# a "safe apply" function that returns infinity (which should never be the minimum of anything) for invalid inputs:

# ????
def safe(f):
    """return a new function that's the same as f, except that it outputs infinity whenever f produces an error"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf') # this means "infinity" in Python
    return safe_f

# Putting It All Together:
# In the general case, we have some target_fn that we want to minimize, and we also have its gradient_fn.
# For example, the target_fn could represent the errors in a model as a function of its parameters,
# and we might want to find the parameters that make the errors as small as possible.
# Furthermore, let's say we have (somehow) chosen a starting value for the parameters theta_0.
# Then we can implement gradient descent as:

# ???? Below can be tested on a function.
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta_0             # set theta to initial value
    target_fn = safe(target_fn) # safe version of target_fn
    value = target_fn(theta)    # value we're minimizing
    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                        for step_size in step_sizes]
        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

# We called it minimize_batch because, for each gradient step, it looks at the entire data set
# (because target_fn returns the error on the whole data set).
# In the next section, we'll see an alternative approach that only looks at one data point at a time.
# Sometimes we'll instead want to maximize a function, which we can do by minimizing its negative
# (which has a corresponding negative gradient):

# ???? Below ones
def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                            negate_all(gradient_fn),
                            theta_0,
                            tolerance)


# Stochastic Gradient Descent:

