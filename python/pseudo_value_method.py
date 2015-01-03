import os, sys

from scipy.stats import norm
import scipy.stats

from scipy.optimize import fminbound

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


def update_mixture_params_estimate_OLD(z1, z2, starting_point):
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
    
    weighted_sum_sqs_1 = (ez*((z1-mu)**2)).sum()
    weighted_sum_sqs_2 = (ez*((z2-mu)**2)).sum()
    weighted_sum_prod = (ez*(z2-mu)*(z1-mu)).sum()

    sigma = math.sqrt((weighted_sum_sqs_1+weighted_sum_sqs_2)/(2*ez_sum))
    #print(weighted_sum_sqs_1, weighted_sum_sqs_2, ez_sum)
    
    rho = 2*(ez*(z1-mu)*(z2-mu)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)
    
    jnt_log_lhd = numpy.log(
        p*numpy.exp(signal_log_lhd) + (1-p)*numpy.exp(noise_log_lhd)).sum()
    return ((mu, mu), (sigma, sigma), rho, p), jnt_log_lhd, jnt_log_lhd

def calc_log_lhd(theta, z1, z2):
    mu, sigma, rho, p = theta
    
    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        mu, mu, sigma, sigma, rho, z1, z2)

    log_lhd = numpy.log(p*numpy.exp(signal_log_lhd)
                        +(1-p)*numpy.exp(noise_log_lhd)).sum()
    
    return log_lhd

def calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
    mu, sigma, rho, p = theta

    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        mu, mu, sigma, sigma, rho, z1, z2)

    # calculate the likelihood ratio for each statistic
    ez = p*numpy.exp(signal_log_lhd)/(
        p*numpy.exp(signal_log_lhd)+(1-p)*numpy.exp(noise_log_lhd))
    ez_sum = ez.sum()

    # startndardize the values
    std_z1 = (z1-mu)/sigma
    std_z2 = (z2-mu)/sigma

    # calculate the weighted statistics - we use these for the 
    # gradient calculations
    weighted_sum_sqs_1 = (ez*(std_z1**2)).sum()
    weighted_sum_sqs_2 = (ez*((std_z2)**2)).sum()
    weighted_sum_prod = (ez*std_z2*std_z1).sum()    

    if fix_mu:
        mu_grad = 0
    else:
        mu_grad = (ez*((std_z1+std_z2)/(1-rho*rho))).sum()

    if fix_sigma:
        sigma_grad = 0
    else:
        sigma_grad = (
            weighted_sum_sqs_1 
            + weighted_sum_sqs_2 
            - 2*rho*weighted_sum_prod )
        sigma_grad /= (1-rho*rho)
        sigma_grad -= 2*ez_sum
        sigma_grad /= sigma

    rho_grad = -rho*(rho*rho-1)*ez_sum + (rho*rho+1)*weighted_sum_prod - rho*(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)
    rho_grad /= (1-rho*rho)*(1-rho*rho)

    p_grad = numpy.exp(signal_log_lhd) - numpy.exp(noise_log_lhd)
    p_grad /= p*numpy.exp(signal_log_lhd)+(1-p)*numpy.exp(noise_log_lhd)
    p_grad = p_grad.sum()
    
    return numpy.array((mu_grad, sigma_grad, rho_grad, p_grad))

def update_mixture_params_estimate(z1, z2, starting_point, 
                                   fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point
    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        return calc_log_lhd(theta, z1, z2)

    def bnd_calc_log_lhd_gradient(theta):
        return calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma)
    
    def clip_theta(theta):
        return theta
    
    prev_lhd = bnd_calc_log_lhd(theta)

    for i in range(10000):
        def bnd_objective(alpha):
            new_theta = clip_theta(
                theta + alpha*bnd_calc_log_lhd_gradient(theta) )
            return -bnd_calc_log_lhd( new_theta )
        
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu
        #grad = bnd_calc_log_lhd_gradient(theta)
        # theta[i] + gradient[i]*alpha > 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 1
        for param_val, grad_val in zip(theta, grad):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, param_val/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, -param_val/grad_val)
        
        # theta[3] + gradient[3]*alpha < 1
        # gradient[3]*alpha < 1 - theta[3]
        # if gradient[2] > 0:
        #     alpha < (1 - theta[3])/gradient[3]
        # elif gradient[2] < 0:
        #     alpha < (theta[3] - 1)/gradient[3]
        for param_val, grad_val in zip(theta[2:], grad[2:]):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, (1-param_val)/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, (param_val-1)/grad_val)
        
        alpha = fminbound(bnd_objective, 0, max_alpha)
        log_lhd = -bnd_objective(alpha)
        if abs(log_lhd-prev_lhd) < EPS:
            return ( ((theta[0], theta[0]), 
                      (theta[1], theta[1]), 
                      rho, p),
                     log_lhd, log_lhd )
        else:
            prev_lhd = log_lhd
            theta = clip_theta(theta + alpha*bnd_calc_log_lhd_gradient(theta))
        
    assert False

def estimate_mixture_params(r1_values, r2_values, params):
    prev_lhd = None
    for i in range(MAX_NUM_EM_ITERS):
        new_params, joint_lhd, other_lhd = update_mixture_params_estimate_OLD(
            r1_values, r2_values, params)
        if prev_lhd != None: 
            print( joint_lhd, joint_lhd-prev_lhd, new_params )
        
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
        z1 = compute_pseudo_values(ranks_1, mu[0], sigma[0], p)
        z2 = compute_pseudo_values(ranks_2, mu[1], sigma[1], p)
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
    for i in range(1):
        params = (1, 1, 0.9, 0.5)
        (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
            10000, params)
        params = ((1,1), (1,1), 0.1, 0.9)
        print( update_mixture_params_estimate(r1_values, r2_values, params) )
        #print( estimate_mixture_params(r1_values, r2_values, params) )
    
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
