# system packages
from __future__ import division
from collections import Counter, defaultdict
from functools import partial
import math, random, csv
import matplotlib.pyplot as plt
import dateutil.parser
import datetime

# my packges
from linear_algebra import shape, get_row, get_column, make_matrix, \
    vector_mean, vector_sum, dot, magnitude, vector_subtract, scalar_multiply
from statistics import correlation, standard_deviation, mean
from probability import inverse_normal_cdf
from gradient_descent import maximize_batch

# -*- coding: cp1252 -*- 
#***************************************************************************************************
# **********************Chapter 10. Working with Data***********************************************
#***************************************************************************************************

#???? Skipping visualization part for time being. Have to complete once install matplotlib or move code to Jupyter Notebook.
# But printing values instead of plots.

"""
Exploring One-Dimensional Data
An obvious first step is to compute a few summary statistics. You'd like to know how many data points you have,the smallest,the
largest, the mean, and the standard deviation. But even these don't necessarily give you a great understanding.A good next step
is to create a histogram, in which you group your data into discrete buckets and count how many points fall into each bucket:
"""
def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

# Defining as can not plot without matplotlib.
def plot_histogram_values(points, bucket_size=10):
    histogram = make_histogram(points, bucket_size)
    print histogram

    
# uniform between -100 and 100
'''
uniform = [100 * random.random() - 100 for _ in range(10000)]
print uniform
plot_histogram(uniform)
'''

"""
# normal distribution with mean 0, standard deviation 50
# ???? First understand inverse_normal_cdf
# normal = [50 * inverse_normal_cdf(random.random()) for _ in range(10000)]
# plot_histogram(normal, 10)
"""


# Exploring 2-Dimensional Data
'''
# For example, consider another fake data set:
def random_normal():
    """returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())
xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]
# If you were to run plot_histogram on ys1 and ys2 you'd get very similar looking plots
# (indeed, both are normally distributed with the same mean and standard deviation).

# Figure 10-2. Histogram of normal
# But each has a very different joint distribution with xs, as shown in Figure 10-3:
plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()
'''

# Many Dimensions
# With many dimensions, you'd like to know how all the dimensions relate to one another. A simple approach is to look at
# the correlation matrix, in which the entry in row i and column j is the correlation between the ith dimension and
# the jth dimension of the data:

def correlation_matrix(data):
    """returns the num_columns x num_columns matrix whose (i, j)th entry
    is the correlation between columns i and j of data"""
    _, num_columns = shape(data)
    def matrix_entry(i, j):
        return correlation(get_column(data, i), get_column(data, j))
    return make_matrix(num_columns, num_columns, matrix_entry)


random.seed(2)
a = [[int(10*random.random()) for i in range(7)] for j in range(6)]
print a
print correlation_matrix([[int(10*random.random()) for i in range(7)] for j in range(6)])


# A more visual approach (if you don't have too many dimensions) is to make a scatterplot matrix (Figure 10-4) showing all
# the pairwise scatterplots. To do that we'll use plt.subplots(), which allows us to create subplots of our chart. 
# We give it the number of rows and the # number of columns, and it returns a figure object (which we won't use) 
# and a two-dimensional array of axes objects (each of which we'll # plot to):  Once matplotlib available ???? 

"""
# _, num_columns = shape(data)
# fig, ax = plt.subplots(num_columns, num_columns)

for i in range(num_columns):
    for j in range(num_columns):
        # scatter column_j on the x-axis vs column_i on the y-axis
        if i != j: ax[i][j].scatter(get_column(data, j), get_column(data, i))
        # unless i == j, in which case show the series name
        else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
        xycoords='axes fraction',
        ha="center", va="center")
        # then hide axis labels except left and bottom charts
        if i < num_columns - 1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)
# fix the bottom right and top left axis labels, which are wrong because their charts only have text in them
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()
"""


# Cleaning and Munging:
'''
def parse_row(input_row, parsers):
    """given a list of parsers (some of which may be None) 
    apply the appropriate one to each element of the input_row"""
    return [parser(value) if parser is not None else value
                for value, parser in zip(input_row, parsers)]
'''
# What if there's bad data? A "float" value that doesn't actually represent a number? We'd usually rather get a None
# than crash our program. We can do this with a helper function:
def try_or_none(f):
    """wraps f to return None if f raises an exception assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none

# after which we can rewrite parse_row to use it:
def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
                for value, parser in zip(input_row, parsers)]


def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
#        print 'a',row
        yield parse_row(row, parsers)

'''
data = []
with open("comma_delimited_stock_prices.csv", "rb") as f:
    reader = csv.reader(f)
    for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
        data.append(line)
print data       
'''

# We could create similar helpers for csv.DictReader. In that case, you'd probably want to supply a dict of parsers
# by field name. For example: ???? to do below one.
def try_parse_field(field_name, value, parser_dict):
    """try to parse value using the appropriate function from parser_dict"""
    parser = parser_dict.get(field_name) # None if no such entry
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value

def parse_dict(input_dict, parser_dict):
    return { field_name : try_parse_field(field_name, value, parser_dict)
                for field_name, value in input_dict.iteritems() }


# Manipulating Data:
data = [
{'closing_price': 102.06,
'date': datetime.datetime(2014, 8, 29, 0, 0),
'symbol': 'AAPL'},
# ...
]
# Conceptually we'll think of them as rows (as in a spreadsheet).
# For instance, suppose we want to know the highest-ever closing price for AAPL
'''
max_aapl_price = max(row["closing_price"] for row in data if row["symbol"] == "AAPL")
print max_aapl_price
'''

# More generally, we might want to know the highest-ever closing price for each stock in our data set. One way to do this is:
# 1. Group together all the rows with the same symbol.
# 2. Within each group, do the same as before:
# group rows by symbol
'''
by_symbol = defaultdict(list)
for row in data:
    by_symbol[row["symbol"]].append(row)
# use a dict comprehension to find the max for each symbol
max_price_by_symbol = { symbol : max(row["closing_price"] for row in grouped_rows)
                                    for symbol, grouped_rows in by_symbol.iteritems() }
'''


''' Skipped untill rescalling ????'''


# Rescaling:
# Table 10-1. Heights and Weights
# Person 	Height (inches)	 	Height (centimeters) 	   Weight
# A 	   63 inches 		       160 cm 			       150 pounds
# B 	   67 inches 		       170.2 cm		           160 pounds
# C 	   70 inches 		       177.8 cm 		       171 pounds

'''
# If we measure height in inches, then B's nearest neighbor is A:
a_to_b = distance([63, 150], [67, 160]) 		# 10.77
a_to_c = distance([63, 150], [70, 171]) 		# 22.14
b_to_c = distance([67, 160], [70, 171]) 		# 11.40
print 'a_to_b, a_to_c and b_to_c',a_to_b,a_to_c,b_to_c

# However, if we measure height in centimeters, then B's nearest neighbor is instead C:
a_to_b = distance([160, 150], [170.2, 160])	 # 14.28
a_to_c = distance([160, 150], [177.8, 171]) 	# 27.53
b_to_c = distance([170.2, 160], [177.8, 171]) 	# 13.37
print 'a_to_b, a_to_c and b_to_c',a_to_b,a_to_c,b_to_c
'''

# Obviously it is problematic if changing units can change results like this. For this reason, when dimensions aren't comparable
# with one another, we will sometimes rescale our data so that each dimension has mean 0 and standard deviation 1.
# This effectively gets rid of the units, converting each dimension to "standard deviations from the mean."

# To start with, we'll need to compute the mean and the standard_deviation for each column:
def scale(data_matrix):
    """returns the means and standard deviations of each column"""
    num_rows,num_columns=la.shape(data_matrix)
    means=[mean(la.get_column(data_matrix,j)) for j in range(num_columns)]
    std_dev=[standard_deviation(get_column(data_matrix,j)) for j in range(num_columns)] 
    return means,std_dev
# print scale([[1,2,4],[3,5,6],[8,1,9]])

# Now use means and std_dev to create a new data matrix:
def rescale(data_matrix):
    """rescales the input data so that each column has mean 0 and standard deviation 1
    leaves alone columns with no deviation"""
    means, stdevs = scale(data_matrix)
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)
# print rescale([[1,2,4],[3,5,6],[8,1,9]])


# Dimensionality Reduction:
# Most of the variation in the data seems to be along a single dimension that doesn't correspond to 
# either the x-axis or the y-axis. 
# When this is the case, we can use a technique called principal component analysis to extract
# one or more dimensions that capture as much of the variation in the data as possible.

# As a first step, we'll need to translate the data so that each dimension has mean zero:
def de_mean_matrix(A):
    """returns the result of subtracting from every value in A the mean value of its column. 
    the resulting matrix has mean 0 in every column"""
    num_rows,num_cols=la.shape(A)      
    col_means,_=scale(A)     # Scale returns 
    return la.make_matrix(num_rows,num_cols,lambda i,j:A[i][j]-col_means[j])
# print de_mean_matrix([[1,2,4],[3,5,6],[8,1,9]])

# Skipped ???? Do later as very important





    


