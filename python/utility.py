import numpy

from scipy.special import erf
from scipy.optimize import brentq

import math

def simulate_values(N, params):
    """Simulate ranks and values from a mixture of gaussians

    """
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

# import the inverse cdf functions
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


def calc_gaussian_lhd(mu_1, mu_2, sigma_1, sigma_2, rho, z1, z2):
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

def calc_post_membership_prbs(theta, z1, z2):
    mu, sigma, rho, p = theta
    
    noise_log_lhd = calc_gaussian_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = calc_gaussian_lhd(
        mu, mu, sigma, sigma, rho, z1, z2)
    
    ez = p*numpy.exp(signal_log_lhd)/(
        p*numpy.exp(signal_log_lhd)+(1-p)*numpy.exp(noise_log_lhd))
    
    return ez

def calc_gaussian_mix_log_lhd(theta, z1, z2):
    mu, sigma, rho, p = theta

    noise_log_lhd = calc_gaussian_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = calc_gaussian_lhd(
        mu, mu, sigma, sigma, rho, z1, z2)

    log_lhd = numpy.log(p*numpy.exp(signal_log_lhd)
                        +(1-p)*numpy.exp(noise_log_lhd)).sum()
    return log_lhd

def calc_gaussian_mix_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
    mu, sigma, rho, p = theta

    noise_log_lhd = calc_gaussian_lhd(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = calc_gaussian_lhd(
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
