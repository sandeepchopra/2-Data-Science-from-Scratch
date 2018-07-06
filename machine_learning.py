from __future__ import division
from collections import Counter
import math, random

# Overfitting and Underfitting:
# The simplest way to do this is to split your data set, so that (for example) two-thirds of it is used to train
# the model, after which we measure the model's performance on the remaining third:
def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []   # Tuple of 2 lists
#    print results
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results
'''
X=[[100*random.random() for i in range(4)] for j in range(10)] # Create 4 * 10 matrix.
for i in X:
    print i
y=[[100*random.random() for i in range(1)] for j in range(10)] 
for i in y:
    print i
a=zip(X,y)
print a
'''

def train_test_split(x, y, test_pct):
    data = zip(x, y)				               # pair corresponding values
    train, test = split_data(data, 1 - test_pct)   # split the data set of pairs
    print train
    print test
    x_train, y_train = zip(*train)		           # magical un-zip trick ???? later
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test
'''
X=[[100*random.random() for i in range(4)] for j in range(10)] # Create 4 * 10 matrix.
y=[[100*random.random() for i in range(1)] for j in range(10)] 
a=train_test_split(X,y,.10)
'''


# Correctness:

# As we'll see below, this test is indeed more than 98% accurate. Nonetheless, it's an incredibly stupid test,
# and a good illustration of why we don't typically use "accuracy" to measure how good a model is.
# We often represent these as counts in a confusion matrix:
#      				        Spam 			    not Spam
# predict "Spam" 		    True Positive 		False Positive
# predict "Not Spam"		False Negative 		True Negative

#              leukemia 	no leukemia 	total
# "Luke" 		70 		    4,930 		    5,000
# not "Luke" 	13,930 		981,070 	    995,000
# total 		14,000 		986,000	        1,000,000

# We can then use these to compute various statistics about model performance. 
# For example ,accuracy is defined as the fraction of correct predictions:
def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total
# print accuracy(70, 4930, 13930, 981070) 	# 0.98114 That's why only accuracy is stupid.

# That seems like a pretty impressive number. But clearly this is not a good test, which means that we probably 
# shouldn't put a lot of credence in raw accuracy.
# It's common to look at the combination of precision and recall. Precision measures how accurate our positive predictions were:
def precision(tp, fp, fn, tn):
    return tp / (tp + fp)
# print precision(70, 4930, 13930, 981070)	 # 0.014

# And recall measures what fraction of the positives our model identified:
def recall(tp, fp, fn, tn):
    return tp / (tp + fn)
# print recall(70, 4930, 13930, 981070) 	               # 0.005

# These are both terrible numbers, reflecting that this is a terrible model.

# Sometimes precision and recall are combined into the F1 score, which is defined as:
def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
# This is the harmonic mean of precision and recall and necessarily lies between them.



