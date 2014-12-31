import os, sys

from scipy.stats import norm
import scipy.stats

import numpy

import math

MAX_NUM_PSUEDO_VAL_ITER = 10000
MAX_NUM_EM_ITERS = 1000
EPS = 1e-6

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

    values = numpy.arange(min_val, max_val, (max_val-min_val)/10000.) #100*len(ranks)))
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

def compute_lhd_2(mu_1, mu_2, sigma_1, sigma_2, rho, z1, z2):
    # -1.837877 = -log(2)-log(pi)
    std_z1 = (z1-mu_1)/sigma_1
    std_z2 = (z2-mu_2)/sigma_2
    loglik = ( 
        -1.837877 
         - math.log(sigma_1) 
         - math.log(sigma_2) 
         - 0.5*math.log(1-rho*rho)
         - ( std_z1**2 - 2*rho*std_z1*std_z2 + std_z2**2 )/(2*(1-rho*rho))
    )
    return loglik

def update_mixture_params_estimate(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        i_mu[0], i_mu[1], i_sigma[0], i_sigma[1], i_rho, z1, z2)
    
    ez = i_p*numpy.exp(signal_log_lhd)/(
        i_p*numpy.exp(signal_log_lhd)+(1-i_p)*numpy.exp(noise_log_lhd))
    
    # just a small optimization
    ez_sum = ez.sum()
    
    p = ez_sum/len(ez)
    
    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    mu = (mu_1 + mu_2)/2
    #mu = 1
    
    weighted_sum_sqs_1 = (ez*((z1-mu)**2)).sum()
    weighted_sum_sqs_2 = (ez*((z2-mu)**2)).sum()
    sigma = math.sqrt((weighted_sum_sqs_1+weighted_sum_sqs_2)/(2*ez_sum))
    #print(weighted_sum_sqs_1, weighted_sum_sqs_2, ez_sum)
    
    rho = 2*(ez*(z1-mu)*(z2-mu)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)
    #rho = 0.9
    #rho = (ez*(z1-mu)*(z2-mu)).sum()/(2*ez_sum)
    #rho = max(0.01, rho)
    #rho = 0.5

    jnt_log_lhd = numpy.log(
        p*numpy.exp(signal_log_lhd) + (1-p)*numpy.exp(noise_log_lhd)).sum()
    return ((mu, mu), (sigma, sigma), rho, p), jnt_log_lhd, jnt_log_lhd

def update_mixture_params_estimate_fixed(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        i_mu[0], i_mu[1], i_sigma[0], i_sigma[1], i_rho, z1, z2)    
    ez = i_p*numpy.exp(signal_log_lhd)/(
        i_p*numpy.exp(signal_log_lhd)+(1-i_p)*numpy.exp(noise_log_lhd))
    ez_sum = ez.sum()
    
    p = ez_sum/len(ez)

    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    mu = (mu_1 + mu_2)/2
    mu_1 = mu_2 = mu
    
    sigma_1, sigma_2, sigma = 1., 1., 1.
    rho = (ez*(z1-mu)*(z2-mu)).sum()/(ez_sum)
    
    noise_log_lhd = compute_lhd(0, 1, 0, z1, z2)
    signal_log_lhd = compute_lhd(mu, sigma, rho, z1, z2)
    jnt_log_lhd = numpy.log(
        p*numpy.exp(signal_log_lhd) + (1-p)*numpy.exp(noise_log_lhd)).sum()
    return ((mu, mu), (sigma, sigma), rho, p), jnt_log_lhd, jnt_log_lhd

def update_mixture_params_estimate_full(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        i_mu[0], i_mu[1], i_sigma[0], i_sigma[1], i_rho, z1, z2)
    
    ez = i_p*numpy.exp(signal_log_lhd)/(
        i_p*numpy.exp(signal_log_lhd)+(1-i_p)*numpy.exp(noise_log_lhd))
    
    # just a small optimization
    ez_sum = ez.sum()
    
    p = ez_sum/len(ez)
    
    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    
    weighted_sum_sqs_1 = (ez*((z1-mu_1)**2)).sum()
    sigma_1 = math.sqrt(weighted_sum_sqs_1/ez_sum)

    weighted_sum_sqs_2 = (ez*((z2-mu_2)**2)).sum()
    sigma_2 = math.sqrt(weighted_sum_sqs_2/ez_sum)
    
    rho = 2*((ez*(z1-mu_1))*(z2-mu_2)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)
    
    jnt_log_lhd = numpy.log(
        i_p*numpy.exp(signal_log_lhd) + (1-i_p)*numpy.exp(noise_log_lhd)).sum()
    print( jnt_log_lhd, ((mu_1, mu_2), (sigma_1, sigma_2), rho, p) )
    
    return ((mu_1, mu_2), (sigma_1, sigma_1), rho, p), jnt_log_lhd, jnt_log_lhd

def estimate_mixture_params(r1_values, r2_values, params):
    prev_lhd = None
    for i in range(MAX_NUM_EM_ITERS):
        new_params, joint_lhd, other_lhd = update_mixture_params_estimate(
            r1_values, r2_values, params)
        #if prev_lhd != None: 
        #    print( joint_lhd, joint_lhd-prev_lhd, new_params )
        
        assert i < 2 or prev_lhd == None or joint_lhd + 10*EPS > prev_lhd, str(
            joint_lhd + 10*EPS-prev_lhd)

        #if max(abs(x-y) for (x,y) in zip(params, new_params)) < 1e-12:
        if prev_lhd != None and abs(joint_lhd - prev_lhd) <  10*EPS:
            noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, r1_values, r2_values)
            signal_log_lhd = compute_lhd_2(
                new_params[0][0], new_params[0][1],
                new_params[1][0], new_params[0][1], 
                new_params[2], 
                r1_values, r2_values)
            ez = signal_log_lhd/(noise_log_lhd + signal_log_lhd + EPS)
            return new_params, joint_lhd, other_lhd, ez.sum()/len(ez)
        
        params = new_params
        prev_lhd = joint_lhd
    
    raise RuntimeError( "Max num iterations exceeded in EM procedure" )

def em_gaussian(ranks_1, ranks_2, params):
    lhds = []
    param_path = []
    max_lhd, max_index = -1e9, -1
    for i in range(MAX_NUM_PSUEDO_VAL_ITER):
        mu, sigma, rho, p = params
        z1 = compute_pseudo_values(ranks_1, mu[0], sigma[1], p)
        z2 = compute_pseudo_values(ranks_2, mu[0], sigma[1], p)
        params, joint_lhd, other_lhd, ez = estimate_mixture_params(
            z1, z2, params)
        print( joint_lhd, ez, params )
        lhds.append(joint_lhd)
        param_path.append(params)
        diff = joint_lhd - max_lhd
        if i > 10 and diff < 1e-4: # or diff < -1:
            return max_lhd, param_path[max_lhd_index]
        elif joint_lhd > max_lhd:
            max_lhd, max_lhd_index = joint_lhd, i

    raise RuntimeError( "Max num iterations exceeded in pseudo val procedure" )
    

def main():
    #params = (mu, sigma, rho, p)
    for i in range(100):
        params = (1, 1, 0.9, 0.5)
        (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
            10000, params)
        params = ((1,1), (1,1), 0.9, 0.5)
        print( estimate_mixture_params(r1_values, r2_values, params) )
    
    return
    params = (1, 1, 0.9, 0.5)
    (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(10000, params)
    import pylab
    #pylab.scatter(r1_values, r2_values)
    #pylab.scatter(r1_ranks, r2_ranks)
    #pylab.show()
    #return
    params = ((1,1), (1,1), 0.1, 0.9)
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
