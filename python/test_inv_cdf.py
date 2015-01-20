import os, sys

import math

import numpy

from scipy.special import erf, erfinv
from scipy.optimize import bisect, brentq

import timeit

try: 
    import pyximport; pyximport.install()
    from inv_cdf import cdf, cdf_i
except ImportError:
    def cdf(x, mu, sigma, pi):
        norm_x = (x-mu)/sigma
        return 0.5*( (1-pi)*erf(0.707106781186547461715*norm_x) 
                 + pi*erf(0.707106781186547461715*x) + 1 )

def cdf_i(x, mu, sigma, pi, lb, ub):
    return brentq(lambda x: cdf(x, mu, sigma, pi) - r, lb, ub)
x
def compute_pseudo_values(ranks, signal_mu, signal_sd, p):
    pseudo_values = []
    for x in ranks:
        new_x = float(x+1)/(len(ranks)+1)
        pseudo_values.append( cdf_i( new_x, signal_mu, signal_sd, p, -10, 10 ) )

    return numpy.array(pseudo_values)


def py_cdf(x, mu, sigma, pi):
    norm_x = (x-mu)/sigma
    return 0.5*( (1-pi)*erf(0.707106781186547461715*norm_x) 
             + pi*erf(0.707106781186547461715*x) + 1 )

def py_cdf_i(r, mu, sigma, pi, lb, ub):
    return brentq(lambda x: cdf(x, mu, sigma, pi) - r, lb, ub)

def py_compute_pseudo_values(ranks, signal_mu, signal_sd, p):
    pseudo_values = []
    for x in ranks:
        new_x = float(x+1)/(len(ranks)+1)
        pseudo_values.append( 
            py_cdf_i( new_x, signal_mu, signal_sd, p, -10, 10 ) )

    return numpy.array(pseudo_values)



def simulate_values(N, params):
    mu, sigma, rho, p = params
    signal_sim_values = numpy.random.multivariate_normal(
        numpy.array((mu,mu)), 
        numpy.array(((sigma,rho), (rho,sigma))), 
        int(N*p) )
    noise_sim_values = numpy.random.multivariate_normal(
        numpy.array((0,0)), 
        numpy.array(((1,0), (0,1))), 
        N - int(N*p) )
    sim_values = numpy.vstack((signal_sim_values, noise_sim_values))
    sim_values = (sim_values[:,0], sim_values[:,1])
    
    return [x.argsort().argsort() for x in sim_values], sim_values


params = (0, 1, 0.0, 0.5)
(r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
    10000, params)

def t1():
    return compute_pseudo_values(r1_ranks, 1, 1, 0.5)

def t2():
    return py_compute_pseudo_values(r1_ranks, 1, 1, 0.5)

print timeit.timeit( "t1()", number=10, setup="from __main__ import t1"  )
print timeit.timeit( "t2()", number=10, setup="from __main__ import t2"  )
