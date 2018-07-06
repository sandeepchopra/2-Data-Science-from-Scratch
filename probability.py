from __future__ import division
from collections import Counter
import math, random

# -*- coding: cp1252 -*-
#***************************************************************************************************
# **********************Chapter 6. Probability***************************************************

# We could also ask about the probability of the event 'both children are girls' conditional on the event 'at least one 
# of the children is a girl' (L). Surprisingly, the answer is different from before!
# As before, the event B and L ('both children are girls and at least one of the children is a girl') is just the event B. 
# This means we have:
# P(B|L)=P(B,L)/P(L)= P(B)/P(L)=1/3
# My solution: 
# P(B/L)=P(L/B)*P(B)/P(L)
# P(L/B)=1 probability at least 1 is girl given both are girls 1.
# P(B)=(1/2)*(1/2)=1/4 Probability that both are girls.
# P(L)=GirlGirl+BoyGirl+GirlBoy=1/4+1/4+1/4=3/4
# So P(B)/P(L)=(1/4)*(4/3)=1/3

#Prove through program
def randon_kid():
    return random.choice(['boy','girl'])
'''
older_girl=0
both_girls=0
either_girl=0
random.seed(0)
for _ in xrange(10000):
    older=randon_kid()
    younger=randon_kid()
    if older=='girl':
        older_girl+=1
    if older=='girl' or younger=='girl':
        either_girl+=1
    if older=='girl' and younger=='girl':
        both_girls+=1
print 'P(both girls/older girl ', both_girls/older_girl
print 'P(both girls/either girl ', both_girls/either_girl
'''

# Expected value:
# We will sometimes talk about the expected value of a random variable, which is the average of its values weighted
# The coin flip variable has an expected value of 1/2 (= 0 * 1/2 + 1 * 1/2),
# by their probabilities. P1*outcome+P2*outcome...Pn*outcome   
# and the range(10) variable has an expected value of 4.5=(.1*0+.1*1...+.1*9)

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x): 
    "returns the probability that a uniform random variable is <= x"
    if x < 0: return 0 	        # uniform random is never less than 0
    elif x < 1: return x 	# e.g. P(X <= 0.4) = 0.4
    else: return 1 		# uniform random is always less than 1


# It tells probability at a particular point. Max p should be at x=0 for mu=0 and sigma=1 as 0 will be mean.
def normal_pdf(x, mu=0, sigma=1):  
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

# It tells probability from start to a particular point. Probability lies with some standard deviations.
# The cumulative distribution function for the normal distribution cannot be written in an
# using Python's math.erf:
def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

# Sometimes we'll need to invert normal_cdf to find the value corresponding to a specified probability.
# There's no simple way to compute its inverse, but normal_cdf is continuous and strictly increasing, so we can use a binary search:
# The function repeatedly bisects intervals until it narrows in on a Z that's close enough to the desired probability.
# To do ????
def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z, low_p = -10.0, 0          # normal_cdf(-10) is (very close to) 0
    hi_z, hi_p = 10.0, 1             # normal_cdf(10) is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2   # consider the midpoint
        mid_p = normal_cdf(mid_z)    # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z


# An easy way to illustrate this is by looking at binomial random variables, which have two parameters n and p. 
# A Binomial(n,p) random variable is simply the sum of n independent Bernoulli(p) random variables, 
# each of which equals 1 with probability p and 0 with probability 1-p:
def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))

# 75 should have most value.
'''
data=Counter([binomial(100, .75) for _ in range(100)])
print data
'''


# ???? Later
def make_hist(p, n, num_points):
    data = [binomial(n, p) for _ in range(num_points)]
    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
            for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()


