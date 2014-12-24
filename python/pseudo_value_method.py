import os, sys

from scipy.stats import norm
import scipy.stats

import numpy

import math

MAX_NUM_PSUEDO_VAL_ITER = 1000
MAX_NUM_EM_ITERS = 1000

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

def compute_pseudo_values(ranks, signal_mu, signal_sd, p):
    noise_norm = norm(0, 1)
    signal_norm = norm(signal_mu, signal_sd)
    
    # build a grid
    min_quantile = 1./(1.+len(ranks))
    max_quantile = len(ranks)/(1.+len(ranks))

    min_val = min(noise_norm.ppf(min_quantile), 
                  signal_norm.ppf(min_quantile) )
    max_val = max(noise_norm.ppf(max_quantile), 
                  signal_norm.ppf(max_quantile))

    values = numpy.arange(min_val, max_val, (max_val-min_val)/1000.) #100*len(ranks)))
    cdf = p*signal_norm.pdf(values) + (1-p)*noise_norm.pdf(values)
    cdf = numpy.hstack((numpy.zeros(1, dtype=float), (cdf/cdf.sum()).cumsum()))
    pseudo_values = []
    for x in ranks:
        i = cdf.searchsorted(float(x)/len(ranks))
        pseudo_values.append(values[i])

    return numpy.array(pseudo_values)

def compute_lhd(mu, sigma, rho, z1, z2):
    # -1.837877 = -log(2)-log(pi)
    coef = -1.837877-2*math.log(sigma) - math.log(1-rho*rho)/2
    loglik = coef-0.5/((1-rho*rho)*(sigma*sigma))*(
        (z1-mu)**2 -2*rho*(z1-mu)*(z2-mu) + (z2-mu)**2)
    return loglik

def update_mixture_params_estimate(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd(0, 1, 0, z1, z2)
    signal_log_lhd = compute_lhd(i_mu, i_sigma, i_rho, z1, z2)
    
    #ez = i_p/(i_p + (1-i_p)*numpy.exp(noise_log_lhd-signal_log_lhd))
    ez = (i_p*numpy.exp(signal_log_lhd))/(
        i_p*numpy.exp(signal_log_lhd)+(1-i_p)*numpy.exp(noise_log_lhd))
    #ez = numpy.exp(signal_log_lhd)/(
    #    numpy.exp(signal_log_lhd)+numpy.exp(noise_log_lhd))
    p  = min(0.9, ez.mean())
    
    #mu = ((z1+z2)*ez).sum()/(2*ez.sum())
    #mu = max(0, mu)
    mu = 0

    #sigma = math.sqrt((ez*((z1-mu)**2+(z2-mu)**2)).sum()/(2*ez.sum()))
    #sigma = max(sigma, 0.2)
    sigma = 1
    
    rho = 2*((ez*(z1-mu)*(z2-mu)).sum()/(ez*((z1-mu)**2+(z2-mu)**2)).sum())
    #rho = max(0.01, rho)
    #rho = 0.5

    jnt_log_lhd = (
        p*numpy.exp(signal_log_lhd) + (1-p)*numpy.exp(noise_log_lhd))
    return (mu, sigma, rho, p), numpy.log(jnt_log_lhd).sum()

def estimate_mixture_params(r1_values, r2_values, params):
    for i in range(MAX_NUM_EM_ITERS):
        new_params, joint_lhd = update_mixture_params_estimate(
            r1_values, r2_values, params)
        if max(abs(x-y) for (x,y) in zip(params, new_params)) < 1e-12:
            return new_params, joint_lhd
        params = new_params
    
    raise RuntimeError( "Max num iterations exceeded in EM procedure" )

def em_gaussian(ranks_1, ranks_2, params):
    lhds = []
    param_path = []
    max_lhd, max_index = -1e9, -1
    for i in range(MAX_NUM_PSUEDO_VAL_ITER):
        mu, sigma, rho, p = params
        z1 = compute_pseudo_values(ranks_1, mu, sigma, p)
        z2 = compute_pseudo_values(ranks_2, mu, sigma, p)
        params, joint_lhd = estimate_mixture_params(z1, z2, params)
        lhds.append(joint_lhd)
        param_path.append(params)
        diff = joint_lhd - max_lhd
        print( i, joint_lhd, params )
        if i > 10 and abs(diff) < 1e-4: # or diff < -1:
            return max_lhd, param_path[max_lhd_index]
        elif joint_lhd > max_lhd:
            max_lhd, max_lhd_index = joint_lhd, i

    raise RuntimeError( "Max num iterations exceeded in pseudo val procedure" )
    

def main():
    #params = (mu, sigma, rho, p)
    params = (1, 1, 0.1, 0.5)
    (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(1000, params)
    params = (1, 0.5, 0.5, 0.1)
    em_gaussian(r1_ranks, r2_ranks, params)
    return
    
    print( estimate_mixture_params(r1_values, r2_values, params) )
    return
    print
    print(sim_values)
    import pylab
    pylab.scatter(sim_values[:,0], sim_values[:,1])
    pylab.show()
    pass

if __name__ == '__main__':
    main()
