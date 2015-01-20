import os, sys

from scipy.stats import norm
import scipy.stats

from scipy.optimize import fminbound

from scipy.special import erf, erfinv
from scipy.optimize import bisect, brentq

import numpy

import math

MAX_NUM_PSUEDO_VAL_ITER = 10000
MAX_NUM_EM_ITERS = 1000
EPS = 1e-6


import symbolic
calc_loss, calc_grad = symbolic.build_squared_gradient_loss()

#(calc_log_lhd_new, calc_log_lhd_gradient_new, GMCDF_i
# ) = symbolic.build_copula_mixture_loss_and_grad()

#(calc_log_lhd_new, calc_log_lhd_gradient_new 
# ) = symbolic.build_standard_mixture_loss_and_grad()


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

try: 
    import pyximport; pyximport.install()
    from inv_cdf import cdf, cdf_i
except ImportError:
    def cdf(x, mu, sigma, pi):
        norm_x = (x-mu)/sigma
        return 0.5*( (1-pi)*erf(0.707106781186547461715*norm_x) 
                 + pi*erf(0.707106781186547461715*x) + 1 )

#def cdf_i(r, mu, sigma, pi, lb, ub):
#    return brentq(lambda x: cdf(x, mu, sigma, pi) - r, lb, ub)

def compute_pseudo_values(ranks, signal_mu, signal_sd, p):
    pseudo_values = []
    for x in ranks:
        new_x = float(x+1)/(len(ranks)+1)
        pseudo_values.append( cdf_i( new_x, signal_mu, signal_sd, p, -10, 10 ) )

    return numpy.array(pseudo_values)

GMCDF_i = compute_pseudo_values

def compute_lhd(mu_1, mu_2, sigma_1, sigma_2, rho, z1, z2):
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

def update_mixture_params_estimate_OLD(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd(
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

    noise_log_lhd = compute_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd(
        mu, mu, sigma, sigma, rho, z1, z2)

    log_lhd = numpy.log(p*numpy.exp(signal_log_lhd)
                        +(1-p)*numpy.exp(noise_log_lhd)).sum()

    return log_lhd

def calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
    mu, sigma, rho, p = theta

    noise_log_lhd = compute_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd(
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

def update_mixture_params_archive(z1, z2, starting_point, 
                                  fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        return calc_log_lhd(theta, z1, z2)

    def bnd_calc_log_lhd_gradient(theta):
        return calc_log_lhd_gradient(
            theta, z1, z2, fix_mu, fix_sigma)
    
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
        gradient = bnd_calc_log_lhd_gradient(theta)
        #gradient = gradient/numpy.abs(gradient).sum()
        #print( gradient )
        def bnd_objective(alpha):
            new_theta = theta + alpha*gradient
            return -bnd_calc_log_lhd( new_theta )
                
        alpha = fminbound(bnd_objective, 0, find_max_step_size(theta))
        log_lhd = -bnd_objective(alpha)
        
        #alpha = min(1e-11, find_max_step_size(theta))
        #while alpha > 0 and log_lhd < prev_lhd:
        #    alpha /= 10
        #    log_lhd = -bnd_objective(alpha)
        
        if abs(log_lhd-prev_lhd) < EPS:
            return ( ((theta[0], theta[0]), 
                      (theta[1], theta[1]), 
                      theta[2], theta[3]), 
                     log_lhd )
        else:
            theta += alpha*gradient
            #print( "\t", log_lhd, prev_lhd, theta )
            prev_lhd = log_lhd
    
    assert False

def grid_search(r1, r2 ):
    #curr_z1 = GMCDF_i(r1, starting_point[0][0], starting_point[1][0], starting_point[3])
    #curr_z2 = GMCDF_i(r2, starting_point[0][0], starting_point[1][0], starting_point[3])

    res = []
    best_theta = None
    max_log_lhd = -1e100
    for mu in numpy.linspace(0.1, 5, num=10):
        for sigma in numpy.linspace(0.5, 3, num=10):
            for rho in numpy.linspace(0.1, 0.9, num=10):
                for pi in numpy.linspace(0.1, 0.9, num=10):
                    z1 = GMCDF_i(r1, mu, sigma, pi)
                    z2 = GMCDF_i(r2, mu, sigma, pi)
                    log_lhd = calc_log_lhd((mu, sigma, rho, pi), z1, z2)
                    if log_lhd > max_log_lhd:
                        best_theta = ((mu,mu), (sigma,sigma), rho, pi)
                        max_log_lhd = log_lhd
    
    return best_theta

def update_mixture_params_estimate_BAD(r1, r2, starting_point, 
                                   fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        rv = calc_log_lhd_new(theta, z1, z2)
        return rv

    def bnd_calc_log_lhd_gradient(theta):
        z1 = GMCDF_i(r1, mu[0], sigma[0], p)
        z2 = GMCDF_i(r2, mu[1], sigma[1], p)
        return calc_log_lhd_gradient_new(
            theta, z1, z2, fix_mu, fix_sigma)
    
    def find_max_step_size(theta, grad):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 10000
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

    inactive_set = []
    inactive_alphas = []
    for i in range(10000):
        gradient = bnd_calc_log_lhd_gradient(theta)
        for index in inactive_set:
            gradient[index] = 0
        gradient[ numpy.abs(gradient) != numpy.abs(gradient).max() ] = 0
        #print( numpy.argmax(numpy.abs(gradient)) )
        current_index = numpy.argmax(numpy.abs(gradient)) 
        
        norm_gradient = gradient/numpy.abs(gradient).sum()
        max_step_size = find_max_step_size(theta, norm_gradient)
        
        def bnd_objective(alpha):
            new_theta = theta + alpha*norm_gradient
            rv = -bnd_calc_log_lhd( new_theta )
            return rv
                
        alpha = fminbound(bnd_objective, 0, max_step_size)
        #alpha = min(1e-2, find_max_step_size(theta))
        log_lhd = -bnd_objective(alpha)
        
        if log_lhd <= prev_lhd:
            inactive_set.append( current_index )
            inactive_alphas.append( alpha )
            if len( inactive_set ) < 4:
                continue
        else:
            theta += alpha*norm_gradient
            prev_lhd = log_lhd

        if len( inactive_set ) == 4:
            print( inactive_set )
            print( inactive_alphas )
            inactive_set, inactive_alphas = [], []
        
        print( log_lhd, log_lhd-prev_lhd, alpha, max_step_size, current_index )
        print( "gradient", bnd_calc_log_lhd_gradient(theta  + alpha*norm_gradient ) )
        print( "params", theta + alpha*norm_gradient )
        print( "="*20 )

        
        if False and abs(log_lhd-prev_lhd) < 10*EPS:            
                return ( theta, log_lhd )
    
    assert False

def update_mixture_params_estimate(r1, r2, starting_point, 
                                   fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        rv = calc_loss(theta, z1, z2)
        return rv

    def bnd_calc_log_lhd_gradient(theta):
        z1 = GMCDF_i(r1, mu[0], sigma[0], p)
        z2 = GMCDF_i(r2, mu[1], sigma[1], p)
        rv = calc_grad(
            theta, z1, z2, fix_mu, fix_sigma)
        return rv
    
    def find_max_step_size(theta, grad):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 100
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

    inactive_set = []
    inactive_alphas = []
    for i in range(10000):
        gradient = bnd_calc_log_lhd_gradient(theta)
        norm_gradient = gradient/(10000*numpy.abs(gradient).sum())
        
        max_step_size = find_max_step_size(theta, norm_gradient)
        
        def bnd_objective(alpha):
            new_theta = theta - alpha*norm_gradient
            rv = bnd_calc_log_lhd( new_theta )
            return rv
                
        alpha = fminbound(bnd_objective, -1e-6, max_step_size)
        #alpha = min(1e-2, find_max_step_size(theta))
        log_lhd = bnd_objective(alpha)
        
        if False and abs(log_lhd-prev_lhd) < 10*EPS:            
            return theta
        else:
            theta -= alpha*norm_gradient
            prev_lhd = log_lhd

        print( log_lhd, log_lhd-prev_lhd, alpha, max_step_size )
        print( "gradient", gradient )
        print( "params", theta )
        print( "="*20 )

        
        if False and abs(log_lhd-prev_lhd) < 10*EPS:            
                return ( theta, log_lhd )
    
    assert False


def estimate_mixture_params(r1_values, r2_values, params):
    prev_lhd = None
    for i in range(MAX_NUM_EM_ITERS):
        #print( "H", i, params )
        new_params, joint_lhd, other_lhd = update_mixture_params_estimate_OLD(
            r1_values, r2_values, params)
        
        assert i < 2 or prev_lhd == None or joint_lhd + 10*EPS > prev_lhd, str(
            joint_lhd + 10*EPS-prev_lhd)

        #if max(abs(x-y) for (x,y) in zip(params, new_params)) < 1e-12:
        if prev_lhd != None and abs(joint_lhd - prev_lhd) <  10*EPS:
            noise_log_lhd = compute_lhd(0,0, 1,1, 0, r1_values, r2_values)
            signal_log_lhd = compute_lhd(
                new_params[0][0], new_params[0][1],
                new_params[1][0], new_params[0][1], 
                new_params[2], 
                r1_values, r2_values)
            ez = signal_log_lhd/(noise_log_lhd + signal_log_lhd + EPS)
            return (new_params[0], 
                    new_params[1], 
                    new_params[2], new_params[3]), joint_lhd
        
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
            params, log_lhd = update_mixture_params_estimate_full(
                z1, z2, params)
        else:
            params, log_lhd = update_mixture_params_archive(
                z1, z2, params) #, fix_mu, fix_sigma)
        
        print( i, log_lhd, params )
        lhds.append(log_lhd)
        param_path.append(params)

        #print( i, end=" ", flush=True) # params, log_lhd
        if prev_lhd != None and abs(log_lhd - prev_lhd) < 1e-4:
            return params, log_lhd
        prev_lhd = log_lhd

    raise RuntimeError( "Max num iterations exceeded in pseudo val procedure" )

def main():
    # mu <- 2.6
    # sigma <- 1.3
    # rho <- 0.8
    # p <- 0.7

    r1_values, r2_values = [], []
    with open(sys.argv[1]) as fp:
        for line in fp:
            r1, r2, _ = line.split()
            r1_values.append(float(r1))
            r2_values.append(float(r2))
    r1_ranks, r2_ranks = [(-numpy.array(x)).argsort().argsort() 
                          for x in (r1_values, r2_values)]
    print( "Finished Loading Data" )

    #starting_point = list(grid_search(r1_ranks, r2_ranks ))
    print( "Finished Grid Search for Starting Point" )
    starting_point = [(0.10000000000000001, 0.10000000000000001), 
                      (0.5, 0.5), 
                      0.90000000000000002, 
                      0.27777777777777779]
    #starting_point = [(0.07, 0.07), (1, 1), 0.87, 0.88]
    print( starting_point )
    #params = (2.6, 1.3, 0.8, 0.7)
    #starting_point = ((params[0],params[0]), 
    #                  (params[1],params[1]), 
    #                  params[2],
    #                  params[3] )

    #theta, log_lhd = em_gaussian(
    #    r1_ranks, r2_ranks, starting_point,
    #    True, False, False)
    #print( "EM", theta, log_lhd)

    theta, log_lhd = update_mixture_params_estimate(
        r1_ranks, r2_ranks, starting_point )
    print( "NEW", theta, log_lhd)
    
    return


if __name__ == '__main__':
    main()
