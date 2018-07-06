from __future__ import division
from collections import Counter
from linear_algebra import sum_of_squares, dot
import math

# -*- coding: cp1252 -*-
#***************************************************************************************************
#*********************************Chapter 5. Statistics*********************************************
#***************************************************************************************************

'''
num_friends    = [100, 49, 41, 40, 25, ... and lots more]
num_points     = len(num_friends) # 204
largest_value  = max(num_friends) # 100
smallest_value = min(num_friends) # 1
sorted_values  = sorted(num_friends)
smallest_value = sorted_values[0] # 1
second_smallest_value = sorted_values[1] # 1
second_largest_value  = sorted_values[-2] # 49
'''

# Central Tendencies------------------->
# 1. Mean: Usually, we'll want some notion of where our data is centered. Most commonly we'll use the mean (or average)
# which is just the sum of the data divided by its count:

def mean(x):
    return sum(x)/len(x)

#2. Median
def median(x):
    total_elements=len(x)
    """NOTE: There are, in fact, nobvious tricks to efficiently compute medians without sorting the data.
    However, they are beyond the scope of this book, so we have to sort the data.
    A generalization of the median is the quantile, which represents the value less than which 
    a certain percentile of the data lies.The median represents the value less than which 50% of the data lies """
    sorted_x=sorted(x)
    if total_elements%2==0:
        return (sorted_x[(total_elements//2-1)]+sorted_x[total_elements//2])/2
    else:
        return sorted_x[total_elements//2]
'''
num_friends_1 = [2, 3, 4, 2, 4]
num_friends_2 = [2, 3, 4, 2, 4,5]
print mean(num_friends_1)
print median(num_friends_1)
print median(num_friends_2)
'''

#3. Quantile. I think can be called percentile as well.
def quantile(x,p):
    """A generalization of the median is the quantile, which represents the value less than 
    which a certain percentile of the data lies. """
    index=int(p*len(x))
    return sorted(x)[index]
'''
num_friends_3 = range(101,201)   
print quantile(num_friends_3,.20)  
'''

#4. Less commonly you might want to look at the mode, or most-common value[s]
from collections import Counter
import random
#random.seed(44)
def mode(x):
    counts=Counter(x)
    max_value=max(counts.values())
    # print 'max_value', max_value
    return [i for i,j in counts.iteritems() if j==max_value]
'''
a=[random.choice(range(1,100)) for _ in range(100)]
print sorted(a)
print mode(a)
'''

# Dispersion------------------->
# Dispersion refers to measures of how spread out our data is. Typically they are statistics for which values near 
# zero signify not spread out at all and for which large values (whatever that means) signify very spread out.
#5. Range: Difference between the largest and smallest elements.
def data_range(x):
    return max(x)-min(x)

#6. Variance: A more complex measure of dispersion is the variance.
def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar=mean(x)
    n=[i-x_bar for i in x]
    # print 'in de_mean x-x_bar is', n
    return n

def variance(x):
    """assumes x has at least two elements"""
    n=len(x)
    mean_deviations=de_mean(x)
    return sum_of_squares(mean_deviations)/n-1

# NOTE
# This looks like it is almost the average squared deviation from the mean, except that we are dividing by n-1 instead of n.
# In fact, when we're dealing with a sample from a larger population, x_bar is only an estimate of the actual mean, 
# which means that on average (x_i-x_bar) ** 2 is an underestimate of x_i's squared deviation from the mean,
# which is why we divide by n-1 instead of n. 
'''
a=[random.choice(range(1,100)) for _ in range(100)]
print variance(a)
'''

#7. Standard Deviation: A more complex measure of dispersion is the variance.
# Now, whatever units our data is in (e.g. 'friends'), all of our measures of central tendency are in that same unit. 
# The range will similarly be in that same unit. The variance, on the other hand, has units that are the square of the original 
# units (e.g 'friends squared').As it can be hard to make sense of these, we often look instead at the standard deviation.

def standard_deviation(x):
    return math.sqrt(variance(x))
a=[random.choice(range(1,100)) for _ in range(100)]
'''
print 'Mean value is {}'.format(mean(a))
print '50 percentile is {}'.format(quantile(a,.50))
print 'standard_deviation is {}'.format(standard_deviation(a))
'''

#8. Interquartile Range
# Both the range and the standard deviation have the same outlier problem that we saw for the mean earlier for the mean.
# Using the same example, if our friendliest user had instead 200 friends, the standard deviation would be 14.89, 
# more than 60% higher! 
# A more robust alternative computes the difference between the 75th percentile value and the 25th percentile value. 
def interquartile_range(x):
    return quantile(x,.75)-quantile(x,.25)
'''
a=[random.choice(range(1,100)) for _ in range(100)]
print 'Interquartile Range is {}'.format(interquartile_range(a))
'''

# Correlation------------------->
# Whereas variance measures how a single variable deviates from its mean, covariance measures how two variables vary 
# in tandem from their means.
# 8. Covariance: measures how two variables vary in tandem from their means
# Recall that dot sums up the products of corresponding pairs of elements. When corresponding elements of x 
# either both above their means or both below their means, a positive number enters the sum. When one is above its 
# mean and the other below, a negative number enters the sum.
# Accordingly, a 'large' positive covariance means that x tends to be large when y is large and small when y is small. 
# A 'large' negative covariance means the opposite - that x tends to be small when y is large and vice versa.
# A covariance close to zero means that no such relationship exists.


def covariance(x,y):
    n=len(x)
    return dot(de_mean(x),de_mean(y))/n-1
               
# 9. Correlation
def correlation(x,y):
    std_x=standard_deviation(x)
    std_y=standard_deviation(y)
    if std_x > 0 and std_y > 0:
        return covariance(x,y)/std_x/std_y
    else:
        return 0
'''
n=15
a=range(1,n)
b=range(1,n)             
c=[-n+i for i in a]
d=[1,3,4,5,6,2,7,8,9,10,11,13,12,14]
print 'a is', a
print 'b is', b
print 'c is', c
print 'd is', d
print 'covariance is: ',covariance(a,b)
print 'correlation is: ',correlation (a,b)                
print 'covariance is: ',covariance(a,c)                
print 'correlation is: ',correlation (a,c)
print 'covariance is: ',covariance(a,c)                
print 'correlation is: ',correlation (a,c)
'''
