from __future__ import division
from probability import normal_cdf, inverse_normal_cdf
import math, random
import matplotlib.pyplot as plt
# %matplotlib inline

# -*- coding: cp1252 -*-
#***************************************************************************************************
#**********************Chapter 7. Hypothesis and Inference****************************************
#***************************************************************************************************

# Example: Flipping a Coin
# Imagine we have a coin and we want to test whether it’s fair. We’ll make the assumption that the coin has some 
# probability p of landing heads, and so our null hypothesis is that the coin is fair — that is, that p=0.5. 
# We will test this against the alternative hypothesis p != .5. In particular, our test will involve flipping the coin 
# some number n times and counting the number of heads X. Each coin flip is a Bernoulli trial, which means 
# that X is a Binomial(n,p) random variable, which (as we saw in Chapter 6) we can approximate using the normal distribution:

def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# the normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# it's above the threshold if it's not below the threshold
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

# it's between if it's less than hi, but not less than lo
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# it's outside if it's not between
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

# We can also do the reverse — find either the nontail region or the (symmetric) interval around the mean that accounts 
# for a certain level of likelihood. For example, if we want to find an interval centered at the mean and containing 
# 60% probability, then we find the cutoffs where the upper and lower tails each contain 20% of the probability (leaving 60%):

# ???? First understand inverse_normal_cdf in previos chapter.
def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds that contain the specified probability"""
    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound

# In particular, let’s say that we choose to flip the coin n=1000 times. If our hypothesis of fairness is true, 
# X should be distributed approximately normally with mean 500 and standard deviation 15.8: 
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
'''
# print "mu_0 which is n*p and sigma_0(SD) which is root[n*p*(1-p)] for 1000 tosses and prob 0.5 are %r and %r" %(mu_0,sigma_0)
# print normal_two_sided_bounds(0.95, mu_0, sigma_0)    O/P is (469.01026640487555, 530.9897335951244)
# print normal_two_sided_bounds(0.95, 0, 1)             O/P is (-1.9599628448486328, 1.9599628448486328)
'''

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # if x is greater than the mean, the tail is what's greater than x
#        print normal_probability_above(x, mu, sigma)
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is what's less than x
        return 2 * normal_probability_below(x, mu, sigma)

# If we were to see 530 heads, we would compute:
# print two_sided_p_value(529.5, mu_0, sigma_0)              # 0.062


# Awesome Simulation: One way to convince yourself that this is a sensible estimate is with a simulation:
# We flip coin 1000 time for 100000 trials and see value of the extreme_value_count / 100000 which gives 
# probability of values out of interval 470 and 530.

'''
extreme_value_count = 0
for _ in range(100000):
    num_heads = sum(1 if random.random() < 0.5 else 0 		# count # of heads
                    for _ in range(1000)) 		        # in 1000 flips
    if num_heads >= 530 or num_heads <= 470: 			# and count how often
        extreme_value_count += 1 			        # the # is 'extreme'
    print extreme_value_count / 100000 # 0.062
'''


# Confidence Intervals
# main confusion comes why 1000 in this math.sqrt(p * (1 - p) / 1000).
# 1000 comes because it shuld be sigma/root(n) for sampling distribution.
# Here we don’t know p, so instead we use our estimate:
'''
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)	  # 0.0158
print sigma
print normal_two_sided_bounds(0.95, mu, sigma) 
'''


# P-hacking: A procedure that erroneously rejects the null hypothesis only 5% of the time will —
# By definition — 5% of the time erroneously reject the null hypothesis:
def run_experiment():
    """flip a fair coin 1000 times, True = heads, False = tails"""
    return [random.random() < .5 for _ in range(1000)]

def reject_fairness(experiment):
    """using the 5% significance levels"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

# random.seed(0)
# experiments = [run_experiment() for _ in range(1000)]
# num_rejections = len([experiment for experiment in experiments if reject_fairness(experiment)])
# print num_rejections # 46

#other way:
def run_experiment1():
    """flip a fair coin 1000 times, True = heads, False = tails"""
    sum1 = sum([random.random() < .5 for _ in range(1000)])
    if (sum1 > 469 and sum1 < 531):
        return 1
    else:
        return 0

#random.seed(0)
#experiments1 = sum(1 - run_experiment1() for _ in range(1000))
#print experiments1


# Example: Running an A/B Test: It is like measuring if 2 samples belong to same population.
def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

# For example, if “tastes great” gets 200 clicks out of 1,000 views and “less bias” gets 180 clicks out of 1,000 views, the statistic equals:
# z = a_b_test_statistic(1000, 200, 1000, 180)	 # -1.14
# The probability of seeing such a large difference if the means were actually equal would be:
# For example, if 'tastes great' gets 200 clicks out of 1,000 views and 'less bias' gets 180 clicks out of 1,000 views, 
# the statistic equals:

"""
# z = a_b_test_statistic(1000, 200, 1000, 180)    # -1.14
# The probability of seeing such a large difference if the means were actually equal would be:
# two_sided_p_value(z)     # 0.254
# which is large enough that you can’t conclude there’s much of a difference. 
"""

# On the other hand, if 'less bias' only got 150 clicks, we would have:
# z = a_b_test_statistic(1000, 200, 1000, 150)    # -2.94
# two_sided_p_value(z)                            # 0.003
# which means there's only a 0.003 probability you would see such a large difference if the ads were equally effective.


# Bayesian Inference:
# To do ????