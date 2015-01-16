import os, sys

from scipy.stats import norm
import scipy.stats

from scipy.optimize import fminbound

import numpy

import math

MAX_NUM_PSUEDO_VAL_ITER = 10000
MAX_NUM_EM_ITERS = 1000
EPS = 1e-6

import sympy
sympy.init_printing(use_unicode=True)

z1_s, z2_s = sympy.symbols("z1, z2")
lamda_s, sigma_s = sympy.symbols("lamda, sigma", positive=True, real=True)
mu_s, rho_s = sympy.symbols("mu, rho", real=True)

std_z1_s = (z1_s - mu_s)/sigma_s
std_z2_s = (z2_s - mu_s)/sigma_s

sym_biv_density = (
                   1./(2.*sympy.pi*sigma_s*sigma_s)
                  )*(
                   1./sympy.sqrt(1.-rho_s**2)
                  )*sympy.exp(-(
                      std_z1_s**2 + std_z2_s**2 - 2*rho_s*std_z1_s*std_z2_s
                  )/(2*(1-rho_s**2)))
sym_signal_density = sym_biv_density
sym_noise_density = sym_biv_density.subs({mu_s:0, rho_s:0, sigma_s:1})

sym_log_lhd = sympy.log(lamda_s*sym_signal_density 
                        + (1-lamda_s)*sym_noise_density)
from sympy.utilities.autowrap import ufuncify
from sympy.printing.theanocode import theano_function

USE_THEANO = True
USE_LAMBDIFY = False
USE_uFUNCIFY = False

numpy.random.seed(0)

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

sym_gradients = []
for sym in (mu_s, sigma_s, rho_s, lamda_s):
    deriv_log_lhd_sym = sympy.diff(sym_log_lhd, sym)
    sym_gradients.append( deriv_log_lhd_sym )

if USE_THEANO:
    theano_gradient = theano_function(
        (mu_s, sigma_s, rho_s, lamda_s, z1_s, z2_s), 
        sym_gradients,
        dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, z1_s: 1, z2_s:1})    

    theano_log_lhd = theano_function(
        (mu_s, sigma_s, rho_s, lamda_s, z1_s, z2_s), 
        [sym_log_lhd,],
        dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, z1_s: 1, z2_s:1})    

if USE_LAMBDIFY:
    lambdify_grad_fns = [
        sympy.lambdify(
                (mu_s, sigma_s, rho_s, lamda_s, z1_s, z2_s), 
                deriv_log_lhd_sym, "numpy")
        for deriv_log_lhd_sym in sym_gradients ]

    lambdify_log_lhd = sympy.lambdify(
        (mu_s, sigma_s, rho_s, lamda_s, z1_s, z2_s), 
        sym_log_lhd, "numpy")

if USE_uFUNCIFY:
    ufuncify_grad_fns = [
        ufuncify(
                (mu_s, sigma_s, rho_s, lamda_s, z1_s, z2_s), 
                deriv_log_lhd_sym)
        for deriv_log_lhd_sym in sym_gradients ]

    ufuncify_log_lhd = ufuncify(
        (mu_s, sigma_s, rho_s, lamda_s, z1_s, z2_s), 
        sym_log_lhd)


def calc_log_lhd_gradient_new(theta, z1, z2, fix_mu, fix_sigma):
    mu, sigma, rho, lamda = theta

    if USE_THEANO:
        res = theano_gradient(
            numpy.repeat(mu, len(z1)),
            numpy.repeat(sigma, len(z1)),
            numpy.repeat(rho, len(z1)),
            numpy.repeat(lamda, len(z1)),
            z1, z2 )
        return numpy.array( [x.sum() for x in res] )
    if USE_LAMBDIFY:
        return numpy.array([grad_fn(mu, sigma, rho, lamda, z1, z2).sum()
                            for grad_fn in lambdify_grad_fns])
    if USE_uFUNCIFY:
        return numpy.array([grad_fn(mu, sigma, rho, lamda, z1, z2).sum()
                            for grad_fn in ufuncify_grad_fns])        
    assert False



def calc_log_lhd_new(theta, z1, z2, fix_mu, fix_sigma):
    mu, sigma, rho, lamda = theta
    if USE_THEANO:
        return theano_log_lhd(
            numpy.repeat(mu, len(z1)),
            numpy.repeat(sigma, len(z1)),
            numpy.repeat(rho, len(z1)),
            numpy.repeat(lamda, len(z1)),
            z1, z2 ).sum()
    if USE_LAMBDIFY:
        return lambdify_log_lhd( mu, sigma, rho, lamda, z1, z2 ).sum()
    if USE_uFUNCIFY:
        return ufuncify_log_lhd( mu, sigma, rho, lamda, z1, z2 ).sum()
    assert False


def update_mixture_params_estimate(z1, z2, starting_point, 
                                   fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        return calc_log_lhd(theta, z1, z2, fix_mu, fix_sigma)

    def bnd_calc_log_lhd_gradient(theta):
        #print( calc_log_lhd_gradient_new(theta, z1, z2, fix_mu, fix_sigma), 
        #       calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma) )
        return calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma)
    
    def find_max_step_size(theta):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu
        grad = bnd_calc_log_lhd_gradient(theta)
        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
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
        
        ## correlation and mix param are less than 1 constraint
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
        
        return max_alpha
    
    prev_lhd = bnd_calc_log_lhd(theta)

    for i in range(10000):
        def bnd_objective(alpha):
            new_theta = theta + alpha*bnd_calc_log_lhd_gradient(theta)
            return -bnd_calc_log_lhd( new_theta )
                
        alpha = fminbound(bnd_objective, 0, find_max_step_size(theta))
        #alpha = min(1e-6, find_max_step_size(theta))
        log_lhd = -bnd_objective(alpha)
        if abs(log_lhd-prev_lhd) < EPS:
            return ( ((theta[0], theta[0]), 
                      (theta[1], theta[1]), 
                      theta[2], theta[3]), 
                     log_lhd )
        else:
            prev_lhd = log_lhd
            theta += alpha*bnd_calc_log_lhd_gradient(theta)
            print( log_lhd, theta )
    
    assert False

def update_mixture_params_estimate_STD(z1, z2, starting_point, 
                                   fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point
    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        print( theta )
        return -calc_log_lhd(theta, z1, z2)

    def bnd_calc_log_lhd_gradient(theta):
        return -calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma)
    
    prev_lhd = bnd_calc_log_lhd(theta)

    from scipy.optimize import minimize
    cons = [(0, 100), (0.10, 10), (0.05, 0.95), (0.05, 0.95)]
    res = minimize(bnd_calc_log_lhd, 
                   theta, 
                   method='TNC',
                   bounds=cons,
                   jac=bnd_calc_log_lhd_gradient,
                   tol=1e-3,
                   options={'maxiter': 10000})
                     #maxfun=100000)
    print( res )
        
    assert False

def estimate_mixture_params(r1_values, r2_values, params):
    prev_lhd = None
    for i in range(MAX_NUM_EM_ITERS):
        new_params, joint_lhd, other_lhd = update_mixture_params_estimate_OLD(
            r1_values, r2_values, params)
        
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
            return new_params, joint_lhd
        
        params = new_params
        prev_lhd = joint_lhd
    
    raise RuntimeError( "Max num iterations exceeded in EM procedure" )

def em_gaussian(ranks_1, ranks_2, params, 
                use_EM=False, fix_mu=False, fix_sigma=False):
    lhds = []
    param_path = []
    prev_lhd = None
    for i in range(MAX_NUM_PSUEDO_VAL_ITER):
        mu, sigma, rho, p = params
        z1 = compute_pseudo_values(ranks_1, mu[0], sigma[0], p)
        z2 = compute_pseudo_values(ranks_2, mu[1], sigma[1], p)
        if use_EM:
            params, log_lhd = estimate_mixture_params(
                z1, z2, params)
        else:
            params, log_lhd = update_mixture_params_estimate(
                z1, z2, params, fix_mu, fix_sigma)

        print( i, log_lhd, params )
        lhds.append(log_lhd)
        param_path.append(params)

        print( i, end=" ", flush=True) # params, log_lhd
        if prev_lhd != None and abs(log_lhd - prev_lhd) < 1e-2:
            return params, log_lhd
        prev_lhd = log_lhd

    raise RuntimeError( "Max num iterations exceeded in pseudo val procedure" )
    

def main():
    #params = (mu, sigma, rho, p)
    for i in range(1):
        params = (1, 1, 0.9, 0.5)
        (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
            10000, params)
        params = ((1,1), (1,1), 0.1, 0.9)
        print(update_mixture_params_estimate(r1_values, r2_values, params))
        print( estimate_mixture_params(r1_values, r2_values, params) )
    
    return
    params = (1, 1, 0.9, 0.5)
    (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(10000, params)
    import pylab
    #pylab.scatter(r1_values, r2_values)
    #pylab.scatter(r1_ranks, r2_ranks)
    #pylab.show()
    #return
    init_params = ((1,1), (1,1), 0.1, 0.9)
    #params, log_lhd = update_mixture_params_estimate(
    #    r1_values, r2_values, init_params)
    #print(params, log_lhd)
    
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, init_params, True)
    print("\nEM", params, log_lhd)
    
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, init_params, 
                                  False, True, True)
    print("\nGA", params, log_lhd)
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, params, 
                                  False, False, True)
    print("\nGA", params, log_lhd)
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, params, 
                                  False, False, False)
    print("\nGA", params, log_lhd)

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
