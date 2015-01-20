cimport cython
from libc.math cimport exp, sqrt, pow, log, erf, abs


@cython.cdivision(True)
cdef double c_cdf(double x, double mu, double sigma, double pi):
    norm_x = (x-mu)/sigma
    return 0.5*(1-pi)*erf(norm_x/sqrt(2)) + 0.5*pi*erf(x/sqrt(2)) + 0.5

@cython.cdivision(True)
def cdf(double x, double mu, double sigma, double pi):
    return c_cdf(x, mu, sigma, pi)

@cython.cdivision(True)
def cdf_i(double r, double mu, double sigma, double pi, 
          double lb, double ub):
    for i in xrange(100):
        mid = lb + (ub - lb)/2.;
        guess = c_cdf(mid, mu, sigma, pi)
        if abs(guess - r) < 1e-6:
            return mid
        elif guess < r:
            lb = mid
        else:
            ub = mid
    #print (lb_val, mid_val, ub_val), (lb, r, ub)
    #assert False
