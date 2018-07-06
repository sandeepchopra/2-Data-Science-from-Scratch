from __future__ import division # want 3 / 2 == 1.5
import math
import matplotlib.pyplot as plt # pyplot
# %matplotlib inline

# -*- coding: cp1252 -*-
#***************************************************************************************************
# **********************Chapter 4. Linear Algebra***************************************************
#***************************************************************************************************

#1. Add Vectors
def vector_add(v,w):
    return [v_i+w_i for v_i,w_i in zip(v,w)]
'''
a=[1,2,3]
b=[4,5,6]
vector_a_sum_b=vector_add(a,b)
print vector_a_sum_b
'''

#2. Subtract Vectors
def vector_subtract(v,w):
    return [v_i-w_i for v_i,w_i in zip(v,w)]

#3. Sum of all vectors
def vector_sum(vectors):
    result=vectors[0]
    for vector in vectors[1:]:
        result=vector_add(result,vector)
    return result
#Above can be written as below as well:
#def vector_sum(vectors):
#    return reduce(vector_add, vectors)

#or even
#vector_sum = partial(reduce, vector_add)
'''
a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
vectors=[a,b,c]
vector_sum_all=vector_sum(vectors)
print vector_sum_all
'''

#4. Multiply vactor by a scaler
def scalar_multiply(c,v):
    """C is a number and v is a vector"""
    return [c*v_i for v_i in v]
#print scalar_multiply.__doc__

#5. Calculate mean of vectors
def vector_mean(vectors):
    """compute the vector whose ith element is the mean of the ith elements of the input vectors"""
    n=len(vectors)
    return scalar_multiply(1/n,vector_sum(vectors))
'''
a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
vectors=[a,b,c]
mean_of_vectors= vector_mean(vectors)
print mean_of_vectors
'''

#6. Dot product of vectors. Another way of saying this is that it's the length of the vector you'd get if you projected v onto w.
def dot(v,w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum([v_i*w_i for v_i,w_i in zip(v,w)])
'''
a=[1,2,3]
b=[4,5,6]
vector_a_dot_b=dot(a,b)
print vector_a_dot_b
'''

#7. sum of squares of vector
def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v,v)
'''
a=[1,2,3]
sum_of_square_vectors=sum_of_squares(a)
print sum_of_square_vectors
'''

#8. it is easy to compute a vector's sum of squares. Which we can use to compute its magnitude (or length)
# import math
def magnitude(v):
    return math.sqrt(sum_of_squares(v))  # math.sqrt is square root function
'''
a=[1,2,3]
magnitude_vectors=magnitude(a)
print magnitude_vectors                          
'''

#9. We now have all the pieces we need to compute the distance between two vectors
def squared_distance(v, w):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(vector_subtract(v,w))
'''
a=[1,2,3]
b=[4,5,6]
vector_squared_distance=squared_distance(a,b)
print vector_squared_distance
'''

#10. distance between 2 vectors
#def distance(v, w):
#    return math.sqrt(squared_distance(v, w))
'''
a=[1,2,3]
b=[4,5,6]
vector_distance=distance(a,b)
print vector_distance
'''

#11. Which is possibly clearer if we write it as (the equivalent):
def distance(v, w):
    return magnitude(vector_subtract(v, w))
'''
a=[1,2,3]
b=[4,5,6]
vector_distance=distance(a,b)
print vector_distance
'''


# Matrices------------------------------------------>

# Because we are representing matrices with Python lists, which are zero-indexed, we will call the first row of a matrix 'row 0'
# and the first column 'column 0'

#1. Get Shape of matrix. Given this list-of-lists representation, the matrix A has len(A) rows and len(A[0]) columns, which we consider its shape.
def shape(A):
    no_rows=len(A)
    no_columns=len(A[0]) if A else 0
    return no_rows,no_columns
'''
A=[[1,2,3],[4,5,6]]
print shape(A)
'''

# If a matrix has n rows and k columns, we will refer to it as a n*k matrix. We can (and sometimes will)
# think of each row of a n*k matrix as a vector of length k, and each column as a vector of length n '''
#2. get the ith row or jth column
def get_row(A,i):
    return A[i]                     # A[i] is already the ith row

def get_column(A,j):
    return [A_i[j]                  # jth element of row A_i
            for A_i in A]           # for each row A_i

# We'll also want to be able to create a matrix given its shape and a function for generating its elements.
# We can do this using a nested list comprehension. we'll see later, we can use an n*k matrix to represent a linear function 
# that maps k-dimensional vectors to nn-dimensional vectors. Several of our techniques and concepts will involve such functions.

#3. Make a matrix
def make_matrix(total_rows,total_columns,entry_fn):
    """returns a num_rows x num_cols matrix whose (i,j)th entry is entry_fn(i, j)"""
    return [[entry_fn(i,j) for j in range(total_columns)] for i in range(total_rows)]

#4. Function to return element to generate matrix
def is_diagonal(i,j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i==j else 0
'''
identity_matrix = make_matrix(5,5,is_diagonal)
print identity_matrix
print make_matrix(5,4,is_diagonal)
'''