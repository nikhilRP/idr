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

VERBOSE = False

#import symbolic
#calc_loss, calc_grad = symbolic.build_squared_gradient_loss()
#full_calc_loss, full_calc_grad = symbolic.build_copula_mixture_loss_and_grad()

#(calc_log_lhd_new, calc_log_lhd_gradient_new
# ) = symbolic.build_copula_mixture_loss_and_grad()

#(calc_log_lhd_new, calc_log_lhd_gradient_new
# ) = symbolic.build_standard_mixture_loss_and_grad()

#def full_calc_loss(theta, z1, z2):
#    return -calc_log_lhd(theta, z1, z2)

def log_lhd_loss(r1, r2, theta):
    mu, sigma, rho, p = theta
    z1 = GMCDF_i(r1, mu, sigma, p)
    z2 = GMCDF_i(r2, mu, sigma, p)
    return -calc_log_lhd(theta, z1, z2)

def sum_grad_sq_loss(r1, r2, theta):
    mu, sigma, rho, p = theta
    z1 = GMCDF_i(r1, mu, sigma, p)
    z2 = GMCDF_i(r2, mu, sigma, p)
    grad = calc_log_lhd_gradient_new(theta, z1, z2, False, False)
    return (grad**2).sum()

calc_loss = log_lhd_loss
#calc_loss = sum_grad_sq_loss

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
        return 0.5*( pi*erf(0.707106781186547461715*norm_x) 
                 + (1-pi)*erf(0.707106781186547461715*x) + 1 )
    def cdf_i(r, mu, sigma, pi, lb, ub):
        return brentq(lambda x: cdf(x, mu, sigma, pi) - r, lb, ub)

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

def EM_estimate(z1, z2, starting_point, fix_mu=False, fix_sigma=False):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd(
        i_mu, i_mu, i_sigma, i_sigma, i_rho, z1, z2)
    
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
    
    rho = 2*(ez*(z1-mu)*(z2-mu)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)

    if fix_mu: mu = i_mu
    if fix_sigma: sigma = i_sigma
    return numpy.array([mu, sigma, rho, p])

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

def grid_search(r1, r2 ):
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

def full_find_max_step_size(theta, grad):
    # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

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

def find_max_step_size(param_val, grad_val, limit_to_1 = False, MIN_VAL=1e-6):
    if grad_val < 0 and param_val < MIN_VAL: return 0
    if limit_to_1 and grad_val > 0 and param_val > MIN_VAL: return 0
    
    max_alpha = 10
    if grad_val > 1e-6:
        max_alpha = min(max_alpha, (param_val-MIN_VAL)/grad_val)
    elif grad_val < -1e-6:
        max_alpha = min(max_alpha, (MIN_VAL-param_val)/grad_val)

    if limit_to_1:
        if grad_val > 1e-6:
            max_alpha = min(max_alpha, (1-param_val-MIN_VAL)/grad_val)
        elif grad_val < -1e-6:
            max_alpha = min(max_alpha, (param_val+MIN_VAL-1)/grad_val)

    return max_alpha    

def coordinate_ascent(r1, r2, theta, gradient_magnitude, 
                      fix_mu=False, fix_sigma=False):
    for j in range(len(theta)):
        if fix_mu and j == 0: continue
        if fix_sigma and j == 1: continue
        
        prev_loss = calc_loss(r1, r2, theta)

        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        real_grad = calc_log_lhd_gradient_new(theta, z1, z2, False, False)
        
        gradient = numpy.zeros(len(theta))
        gradient[j] = gradient_magnitude
        if real_grad[j] < 0: gradient[j] = -gradient[j]
        
        """
        # find the direction of the gradient
        gradient = numpy.zeros(len(theta))
        gradient[j] = gradient_magnitude
        init_alpha = 5e-12
        while init_alpha < 1e-2:
            pos = calc_loss( r1, r2, theta - init_alpha*gradient )
            neg = calc_loss( r1, r2, theta + init_alpha*gradient )
            if neg < prev_loss < pos:
                gradient[j] = gradient[j]
                #assert(calc_loss( r1, r2, theta - init_alpha*gradient ) > prev_loss)
                #assert(calc_loss( r1, r2, theta + init_alpha*gradient ) <= prev_loss)
                break
            elif neg > prev_loss > pos:
                gradient[j] = -gradient[j]
                #assert(calc_loss( r1, r2, theta - init_alpha*gradient ) > prev_loss)
                #assert(calc_loss( r1, r2, theta + init_alpha*gradient ) <= prev_loss)
                break
            else:
                init_alpha *= 10         
        #print( pos - prev_loss, neg - prev_loss )
        assert init_alpha < 1e-1
        """
        
        min_step = 0
        max_step = find_max_step_size(
            theta[j], gradient[j], (False if j in (0,1) else True))

        if max_step < 1e-12: continue

        alpha = fminbound(
            lambda x: calc_loss( r1, r2, theta + x*gradient ),
            min_step, max_step)
        
        
        loss = calc_loss( r1, r2, theta + alpha*gradient )
        #print( "LOSS:", loss, prev_loss, loss-prev_loss )
        if loss < prev_loss:
            theta += alpha*gradient

    return theta

def find_local_maximum_CA(r1, r2, theta, 
                          fix_mu=False, fix_sigma=False ):
    gradient_magnitude = 1e-2
    for i in range(100):
        prev_loss = calc_loss(r1, r2, theta)
        
        # coordiante ascent step
        theta = coordinate_ascent( r1, r2, theta, gradient_magnitude,
                                   fix_mu=fix_mu, fix_sigma=fix_sigma)

        curr_loss = calc_loss(r1, r2, theta)
        if VERBOSE:
            print( "CA%i\t" % i, 
                   "%.2e" % gradient_magnitude, 
                   "%.2e" % (curr_loss-prev_loss), 
                   "%.8f\t" % curr_loss,
                   "%.8f\t" % log_lhd_loss(r1, r2, theta),
                   theta)

        # find the em estimate 
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        em_theta = EM_estimate(z1, z2, theta )

        for j in (3,2):
            tmp_theta = theta.copy()
            tmp_theta[j] = em_theta[j]
            if calc_loss(r1, r2, tmp_theta) < curr_loss:
                theta[j] = em_theta[j]
        
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        grad = calc_log_lhd_gradient_new(theta, z1, z2, False, False)
        #print( "GRAD", grad )

        if abs(curr_loss-prev_loss) < 1e-12:
            if gradient_magnitude > 1e-6:
                gradient_magnitude /= 3
            else:
                return ( theta, curr_loss )
        else:
            gradient_magnitude = min(1e-2, gradient_magnitude*10)
        
    return theta, curr_loss

def EMP_with_pseudo_value_algorithm(
        r1, r2, theta_0, N=100, EPS=1e-4, 
        fix_mu=False, fix_sigma=False):
    
    prev_theta = theta_0
    for i in range(N):
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)            
        prev_theta = theta
        theta = EM_estimate(
            z1, z2, prev_theta, fix_mu=fix_mu, fix_sigma=fix_sigma)
                            
        if VERBOSE:
            print( "Iter %i\t" % i, 
                   "%.2e" % numpy.abs(theta - prev_theta).sum(),
                   "%.4e" % log_lhd_loss(r1, r2, theta),
                   theta)
        if i > 3 and numpy.abs(theta - prev_theta).sum() < EPS: 
            break
        else:
            prev_theta = theta
    
    return theta


def estimate_model_params(
        r1, r2, theta_0, N=100, EPS=1e-4, 
        fix_mu=False, fix_sigma=False):

    #starting_point = list(grid_search(r1_ranks, r2_ranks ))
    #if VERBOSE:
    #    print( "Finished Grid Search for Starting Point" )

    EM_theta, EM_loss = take_EM_steps(
        r1, r2, theta, N=20, fix_mu=fix_mu, fix_sigma=fix_sigma)
    if VERBOSE:
        print( "EM", EM_theta, EM_loss )
    theta, loss = find_local_maximum_CA( 
        r1, r2, EM_theta, fix_mu=fix_mu, fix_sigma=fix_sigma )
    if VERBOSE:
        print( "CA", theta, loss )
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        real_grad = calc_log_lhd_gradient_new(theta, z1, z2, False, False)
        print( "GRAD", real_grad )
    
    return theta, loss

def main():
    r1_values, r2_values = [], []
    with open(sys.argv[1]) as fp:
        for line in fp:
            r1, r2, _ = line.split()
            r1_values.append(float(r1))
            r2_values.append(float(r2))
    r1_ranks, r2_ranks = [(-numpy.array(x)).argsort().argsort() 
                          for x in (r1_values, r2_values)]
    print( "Finished Loading Data" )
    
    params = (2.6, 1.3, 0.8, 0.7)
    params = (2.3634337,   0.55702308,  0.77529659,  0.5945655) 
    params = (0.3634337,   1,  .77529659,  0.945655) 
    starting_point = numpy.array( params )
    theta, log_lhd = estimate_model_params(
        r1_ranks, r2_ranks, starting_point, 
        fix_mu=False, fix_sigma=False)
        
    print( "NEW", theta, log_lhd)
    
    return


if __name__ == '__main__':
    main()
