import os, sys

import numpy

import sympy
import sympy.stats
from sympy.utilities.autowrap import ufuncify
from sympy.printing.theanocode import theano_function

USE_THEANO = True
USE_LAMBDIFY = False
USE_uFUNCIFY = False

#sympy.init_printing(use_unicode=True)
from sympy.printing.mathml import print_mathml, mathml

import scipy.special

try: 
    import cPickle as pickle
except ImportError:
    import pickle

def cdf_and_inv_cdf_gen(mu, sigma, pi, min_val=-100, max_val=100):
    def cdf(x):
        norm_x = (x-mu)/sigma
        return 0.5*(1-pi)*erf(norm_x/math.sqrt(2)) + 0.5*pi*erf(x/math.sqrt(2)) + 0.5
    
    def inv_cdf(r, start=min_val, stop=max_val):
        assert r > 0 and r < 1
        return brentq(lambda x: cdf(x) - r, min_val, max_val)
    
    return cdf, inv_cdf


def compute_pseudo_values(ranks, signal_mu, signal_sd, p):
    cdf, inv_cdf = cdf_and_inv_cdf_gen(signal_mu, signal_sd, p, -20, 20)
    pseudo_values = []
    for x in ranks:
        new_x = float(x+1)/(len(ranks)+1)
        pseudo_values.append( inv_cdf( new_x ) )

    return numpy.array(pseudo_values)

def build_standard_mixture_loss_and_grad():
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



    def calc_log_lhd_new(theta, z1, z2):
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

    return calc_log_lhd_new, calc_log_lhd_gradient_new

def build_copula_mixture_loss_and_grad_natural_params():
    class GaussianPDF(sympy.Function):
        nargs = 3
        is_commutative = False

        @classmethod
        def eval(cls, x, eta1, eta2):
            std_x = eta2*x - eta1
            return ((
                eta2/(sympy.sqrt(sympy.pi*2))
            )*sympy.exp(-(std_x**2)/2))

    class GaussianMixturePDF(sympy.Function):
        nargs = 4

        @classmethod
        def eval(cls, x, eta1, eta2, lamda):
            return lamda*GaussianPDF(x, 0, 1) + (1-lamda)*GaussianPDF(x, eta1, eta2)

    class GaussianMixtureCDF(sympy.Function):
        nargs = 4
        
        @classmethod
        def eval(cls, x, eta1, eta2, lamda):
            z = sympy.symbols("z", real=True, finite=True)
            rv = sympy.simplify(sympy.Integral(
                    GaussianMixturePDF(z, eta1, eta2, lamda), 
                    (z, -sympy.oo, x)).doit(conds='none'))
            return rv

    def calc_GaussianMixtureCDF_inverse(rs, eta1, eta2, lamda):
        res = []
        z = sympy.symbols("z", real=True, finite=True)
        for r in rs:
            assert r > 0 and r < 1
            sp = sympy.mpmath.erfinv(2*r-1) - lamda*(eta1)
            res.append( float(sympy.nsolve(
                (GaussianMixtureCDF(z, eta1, eta2, lamda)-r,), (z,), sp)[0]) )
        return numpy.array(res)
    
    class GaussianMixtureCDF_inverse(sympy.Function):
        def _eval_is_real(self):
            return True

        def _eval_is_finite(self):
            return True

        def fdiff(self, argindex):
            r, eta1, eta2, lamda = self.args
            # if mu=0 and sigma=1, then this is
            # just the inverse standard erf so return erfi
            if eta1 == 0 and eta2 == 1:
                return sympy.diff(sympy.erfi(r), self.args[argindex-1])

            tmp = sympy.symbols("tmp", real=True, finite=True)
            z_s = GaussianMixtureCDF(tmp, eta1, eta2, lamda)
            inv_diff = sympy.diff(z_s, self.args[argindex-1])
            return sympy.simplify(1/inv_diff.subs(tmp, self))            
    
    lamda_s, eta1_s = sympy.symbols(
        "lamda, eta2", positive=True, real=True, finite=True)
    eta2_s, rho_s = sympy.symbols(
        "eta1, rho", real=True, finite=True)
    
    #sigma_s = 1
    
    r1_s = sympy.symbols("r1_s", real=True, finite=True, positive=True)
    r2_s = sympy.symbols("r2_s", real=True, finite=True, positive=True)
    
    ### build the marginal densities
    z_s = sympy.symbols('z_s')
    std_z_s = eta2_s*z_s - eta1_s

    z1_s = GaussianMixtureCDF_inverse(r1_s, eta1_s, eta2_s, lamda_s)
    #z1_s = sympy.Function('z1_s', real=True, finite=True)(
    #    r1_s, mu_s, sigma_s, lamda_s)
    std_z1_s = std_z_s.subs(z_s, z1_s)

    z2_s = GaussianMixtureCDF_inverse(r2_s, eta1_s, eta2_s, lamda_s)
    #z2_s = sympy.Function('z2_s', real=True, finite=True)(
    #    r2_s, mu_s, sigma_s, lamda_s)
    std_z2_s = std_z_s.subs(z_s, z2_s)


    ### build the parametric bivariate normal density
    sym_signal_density = (
                       eta2_s**2/(2.*sympy.pi)
                      )*(
                       1./sympy.sqrt(1.-rho_s**2)
                      )*sympy.exp(-(
                          std_z1_s**2 + std_z2_s**2 - 2*rho_s*std_z1_s*std_z2_s
                      )/(2*(1-rho_s**2)))

    sym_noise_density = (
                       1./(2.*sympy.pi)
                      )*sympy.exp(-(z1_s**2 + z2_s**2)/2)

    # replace the inverse CDF calls with pseudo value symbols
    pv_1, pv_2 = sympy.symbols('pv_1 pv_2', real=True, finite=True)
    
    sym_log_lhd = sympy.simplify(sympy.log(lamda_s*sym_signal_density 
                                           + (1-lamda_s)*sym_noise_density))
    pv_sym_log_lhd = sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2})

    #sympy.pprint( pv_sym_log_lhd )

    #sympy.pprint(sympy.simplify(sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2, mu_s: 0, sigma_s: 1})))
    #sympy.pprint(sympy.simplify(sym_log_lhd.subs({mu_s: 0, sigma_s: 1}).diff(lamda_s)))
    #assert False

    sym_gradients = []
    for sym in (eta1_s, eta2_s, rho_s, lamda_s):
        sym_grad = sympy.diff(sym_log_lhd, sym)
        pv_sym_grad = sym_grad.subs({z1_s: pv_1, z2_s: pv_2})
        #sympy.pprint( pv_sym_grad )
        #print( "="*100 )
        #print( "\n"*10 )
        #print( "="*100 )
        sym_gradients.append( pv_sym_grad )

    if USE_THEANO:
        theano_gradient = theano_function(
            (eta1_s, eta2_s, rho_s, lamda_s, pv_1, pv_2), 
            sym_gradients,
            dims={eta1_s:1, eta2_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})    

        theano_log_lhd = theano_function(
            (eta1_s, eta2_s, rho_s, lamda_s, pv_1, pv_2), 
            [pv_sym_log_lhd,],
            dims={eta1_s:1, eta2_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})    

        
    if USE_LAMBDIFY:
        lambdify_log_lhd = sympy.lambdify(
            (eta1_s, eta2_s, rho_s, lamda_s, pv_1, pv_2), 
            pv_sym_log_lhd, numpy)

        lambdify_grad_fns = []
        for deriv_log_lhd_sym in sym_gradients:
            f = sympy.lambdify(
                (eta1_s, eta2_s, rho_s, lamda_s, pv_1, pv_2), 
                deriv_log_lhd_sym, ["numpy", {'erf': scipy.special.erf}])
            lambdify_grad_fns.append( f )
    
    def calc_log_lhd(theta, z1, z2):
        eta1, eta2, rho, lamda = theta
        
        if USE_THEANO:
            return theano_log_lhd(
                numpy.repeat(eta1, len(z1)),
                numpy.repeat(eta2, len(z1)),
                numpy.repeat(rho, len(z1)),
                numpy.repeat(lamda, len(z1)),
                z1, z2 ).sum()
        if USE_LAMBDIFY:
            return lambdify_log_lhd( eta1, eta2, rho, lamda, z1, z2 ).sum()
        if USE_uFUNCIFY:
            return ufuncify_log_lhd( eta1, eta2, rho, lamda, z1, z2 ).sum()
        assert False
    
    def calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
        eta1, eta2, rho, lamda = theta

        if USE_THEANO:
            res = theano_gradient(
                numpy.repeat(eta1, len(z1)),
                numpy.repeat(eta2, len(z1)),
                numpy.repeat(rho, len(z1)),
                numpy.repeat(lamda, len(z1)),
                z1, z2 )
            return numpy.array( [x.sum() for x in res] )

        if USE_LAMBDIFY:
            res = []
            for grad_fn in lambdify_grad_fns:
                res.append(grad_fn(eta1, eta2, rho, lamda, z1, z2).sum())
            return numpy.array(res)

        if USE_uFUNCIFY:
            return numpy.array([grad_fn(eta1, eta2, rho, lamda, z1, z2).sum()
                                for grad_fn in ufuncify_grad_fns])        
        assert False
    
    return calc_log_lhd, calc_log_lhd_gradient, calc_GaussianMixtureCDF_inverse

def build_copula_mixture_loss_and_grad():

    class GaussianPDF(sympy.Function):
        nargs = 3
        is_commutative = False

        @classmethod
        def eval(cls, x, mu, sigma):
            std_x = (x - mu)/sigma    
            return ((
                1/(sigma*sympy.sqrt(sympy.pi*2))
            )*sympy.exp(-(std_x**2)/2))

    class GaussianMixturePDF(sympy.Function):
        nargs = 4

        @classmethod
        def eval(cls, x, mu, sigma, lamda):
            return (1-lamda)*GaussianPDF(x, 0, 1) + lamda*GaussianPDF(x, mu, sigma)

    class GaussianMixtureCDF(sympy.Function):
        nargs = 4

        res_str = "lamda*erf(sqrt(2)*tmp/2)/2 + lamda*erf(sqrt(2)*(mu - tmp)/(2*sigma))/2 - erf(sqrt(2)*(mu - tmp)/(2*sigma))/2 + 1/2"
        
        @classmethod
        def eval(cls, x, mu, sigma, lamda):
            try:
                assert False
                with open("cached_gaussian_mix_cdf.obj") as fp:
                    rv = pickle.load(fp)
            except:
                z = sympy.symbols("z", real=True, finite=True)
                rv = sympy.simplify(sympy.Integral(
                        GaussianMixturePDF(z, mu, sigma, lamda), 
                        (z, -sympy.oo, x)).doit())
                #with open("cached_gaussian_mix_cdf.obj", "w") as fp:
                #    rv = pickle.dump(rv, fp)                
            return rv

    def calc_GaussianMixtureCDF_inverse(rs, mu, sigma, lamda):
        res = []
        z = sympy.symbols("z", real=True, finite=True)
        for r in rs:
            assert r > 0 and r < 1
            sp = sympy.mpmath.erfinv(2*r-1) - lamda*(mu/sigma)
            res.append( float(sympy.nsolve(
                (GaussianMixtureCDF(z, mu, sigma, lamda)-r,), (z,), sp)[0]) )
        return numpy.array(res)
    
    class GaussianMixtureCDF_inverse(sympy.Function):
        """
        @classmethod
        def eval(cls, r, mu, sigma, lamda):
            if mu == 0 and sigma == 1:
                return sympy.erfi(r)
            return sympy.Function('GaussianMixtureCDF_inverse')(
                r, mu, sigma, lamda)
        """
        def _eval_is_real(self):
            return True

        def _eval_is_finite(self):
            return True

        def fdiff(self, argindex):
            r, mu, sigma, lamda = self.args
            # if mu=0 and sigma=1, then this is
            # just the inverse standard erf so return erfi
            if mu == 0 and sigma == 1:
                return sympy.diff(sympy.erfi(r), self.args[argindex-1])

            tmp = sympy.symbols("tmp", real=True, finite=True)
            z_s = GaussianMixtureCDF(tmp, mu, sigma, lamda)
            inv_diff = sympy.diff(z_s, self.args[argindex-1])
            return sympy.simplify(1/inv_diff.subs(tmp, self))            
    
    lamda_s, sigma_s = sympy.symbols(
        "lamda, sigma", positive=True, real=True, finite=True)
    mu_s, rho_s = sympy.symbols(
        "mu, rho", real=True, finite=True)
    
    #sigma_s = 1
    
    r1_s = sympy.symbols("r1_s", real=True, finite=True, positive=True)
    r2_s = sympy.symbols("r2_s", real=True, finite=True, positive=True)
    
    ### build the marginal densities
    z_s = sympy.symbols('z_s')
    std_z_s = (z_s - mu_s)/sigma_s

    z1_s = GaussianMixtureCDF_inverse(r1_s, mu_s, sigma_s, lamda_s)
    #z1_s = sympy.Function('z1_s', real=True, finite=True)(
    #    r1_s, mu_s, sigma_s, lamda_s)
    std_z1_s = std_z_s.subs(z_s, z1_s)

    z2_s = GaussianMixtureCDF_inverse(r2_s, mu_s, sigma_s, lamda_s)
    #z2_s = sympy.Function('z2_s', real=True, finite=True)(
    #    r2_s, mu_s, sigma_s, lamda_s)
    std_z2_s = std_z_s.subs(z_s, z2_s)


    ### build the parametric bivariate normal density
    sym_signal_density = (
                       1./(2.*sympy.pi*sigma_s*sigma_s)
                      )*(
                       1./sympy.sqrt(1.-rho_s**2)
                      )*sympy.exp(-(
                          std_z1_s**2 + std_z2_s**2 - 2*rho_s*std_z1_s*std_z2_s
                      )/(2*(1-rho_s**2)))

    sym_noise_density = (
                       1./(2.*sympy.pi)
                      )*sympy.exp(-(z1_s**2 + z2_s**2)/2)

    # replace the inverse CDF calls with pseudo value symbols
    pv_1, pv_2 = sympy.symbols('pv_1 pv_2', real=True, finite=True)
    
    sym_log_lhd = sympy.simplify(sympy.log(lamda_s*sym_signal_density 
                                           + (1-lamda_s)*sym_noise_density))
    pv_sym_log_lhd = sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2})

    #sympy.pprint(sympy.simplify(sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2, mu_s: 0, sigma_s: 1})))
    #sympy.pprint(sympy.simplify(sym_log_lhd.subs({mu_s: 0, sigma_s: 1}).diff(lamda_s)))
    #assert False

    sym_gradients = []
    for sym in (mu_s, sigma_s, rho_s, lamda_s):
        sym_grad = sympy.diff(sym_log_lhd, sym)
        pv_sym_grad = sym_grad.subs({z1_s: pv_1, z2_s: pv_2})
        sym_gradients.append( pv_sym_grad )

    if USE_THEANO:
        theano_gradient = theano_function(
            (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
            sym_gradients,
            dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})    

        theano_log_lhd = theano_function(
            (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
            [pv_sym_log_lhd,],
            dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})    

        
    if USE_LAMBDIFY:
        lambdify_log_lhd = sympy.lambdify(
            (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
            pv_sym_log_lhd, numpy)

        lambdify_grad_fns = []
        for deriv_log_lhd_sym in sym_gradients:
            f = sympy.lambdify(
                (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
                deriv_log_lhd_sym, ["numpy", {'erf': scipy.special.erf}])
            lambdify_grad_fns.append( f )
    
    def calc_log_lhd(theta, z1, z2):
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
    
    def calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
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
            res = []
            for grad_fn in lambdify_grad_fns:
                res.append(grad_fn(mu, sigma, rho, lamda, z1, z2).sum())
            return numpy.array(res)

        if USE_uFUNCIFY:
            return numpy.array([grad_fn(mu, sigma, rho, lamda, z1, z2).sum()
                                for grad_fn in ufuncify_grad_fns])        
        assert False
    
    return calc_log_lhd, calc_log_lhd_gradient

def build_squared_gradient_loss():
    try:
        with open("cached_fn.obj", "rb") as fp:
            theano_loss, theano_gradient = pickle.load(fp)
            print( theano_loss, theano_gradient )
    except:    
        print( "CACHE FAILED" )
        class GaussianPDF(sympy.Function):
            nargs = 3
            is_commutative = False

            @classmethod
            def eval(cls, x, mu, sigma):
                std_x = (x - mu)/sigma    
                return ((
                    1/(sigma*sympy.sqrt(sympy.pi*2))
                )*sympy.exp(-(std_x**2)/2))

        class GaussianMixturePDF(sympy.Function):
            nargs = 4

            @classmethod
            def eval(cls, x, mu, sigma, lamda):
                return lamda*GaussianPDF(x, 0, 1) + (1-lamda)*GaussianPDF(x, mu, sigma)

        class GaussianMixtureCDF(sympy.Function):
            nargs = 4

            res_str = "lamda*erf(sqrt(2)*tmp/2)/2 + lamda*erf(sqrt(2)*(mu - tmp)/(2*sigma))/2 - erf(sqrt(2)*(mu - tmp)/(2*sigma))/2 + 1/2"

            @classmethod
            def eval(cls, x, mu, sigma, lamda):
                z = sympy.symbols("z", real=True, finite=True)
                rv = sympy.simplify(sympy.Integral(
                        GaussianMixturePDF(z, mu, sigma, lamda), 
                        (z, -sympy.oo, x)).doit(conds='none'))
                return rv

        def calc_GaussianMixtureCDF_inverse(rs, mu, sigma, lamda):
            res = []
            z = sympy.symbols("z", real=True, finite=True)
            for r in rs:
                assert r > 0 and r < 1
                sp = sympy.mpmath.erfinv(2*r-1) - lamda*(mu/sigma)
                res.append( float(sympy.nsolve(
                    (GaussianMixtureCDF(z, mu, sigma, lamda)-r,), (z,), sp)[0]) )
            return numpy.array(res)

        class GaussianMixtureCDF_inverse(sympy.Function):
            """
            @classmethod
            def eval(cls, r, mu, sigma, lamda):
                if mu == 0 and sigma == 1:
                    return sympy.erfi(r)
                return sympy.Function('GaussianMixtureCDF_inverse')(
                    r, mu, sigma, lamda)
            """
            def _eval_is_real(self):
                return True

            def _eval_is_finite(self):
                return True

            def fdiff(self, argindex):
                r, mu, sigma, lamda = self.args
                # if mu=0 and sigma=1, then this is
                # just the inverse standard erf so return erfi
                if mu == 0 and sigma == 1:
                    return sympy.diff(sympy.erfi(r), self.args[argindex-1])

                tmp = sympy.symbols("tmp", real=True, finite=True)
                z_s = GaussianMixtureCDF(tmp, mu, sigma, lamda)
                inv_diff = sympy.diff(z_s, self.args[argindex-1])
                return sympy.simplify(1/inv_diff.subs(tmp, self))            

        lamda_s = sympy.symbols( "lamda_s", 
                                 positive=True, real=True, finite=True )

        sigma_s = sympy.symbols(
            "sigma", positive=True, real=True, finite=True)
        mu_s, rho_s = sympy.symbols(
            "mu, rho", real=True, finite=True)        

        #sigma_s = 1

        r1_s = sympy.symbols("r1_s", real=True, finite=True, positive=True)
        r2_s = sympy.symbols("r2_s", real=True, finite=True, positive=True)

        ### build the marginal densities
        z_s = sympy.symbols('z_s')
        std_z_s = (z_s - mu_s)/sigma_s

        z1_s = GaussianMixtureCDF_inverse(r1_s, mu_s, sigma_s, lamda_s)
        #z1_s = sympy.Function('z1_s', real=True, finite=True)(
        #    r1_s, mu_s, sigma_s, lamda_s)
        std_z1_s = std_z_s.subs(z_s, z1_s)

        z2_s = GaussianMixtureCDF_inverse(r2_s, mu_s, sigma_s, lamda_s)
        #z2_s = sympy.Function('z2_s', real=True, finite=True)(
        #    r2_s, mu_s, sigma_s, lamda_s)
        std_z2_s = std_z_s.subs(z_s, z2_s)

        
        ### build the parametric bivariate normal density
        sym_signal_density = (
                           1./(2.*sympy.pi*sigma_s*sigma_s)
                          )*(
                           1./sympy.sqrt(1.-rho_s**2)
                          )*sympy.exp(-(
                              std_z1_s**2 + std_z2_s**2 - 2*rho_s*std_z1_s*std_z2_s
                          )/(2*(1-rho_s**2)))

        sym_noise_density = (
                           1./(2.*sympy.pi)
                          )*sympy.exp(-(z1_s**2 + z2_s**2)/2)

        # replace the inverse CDF calls with pseudo value symbols
        pv_1, pv_2 = sympy.symbols('pv_1 pv_2', real=True, finite=True)

        sym_log_lhd = sympy.simplify(sympy.log(lamda_s*sym_signal_density 
                                               + (1-lamda_s)*sym_noise_density))
        pv_sym_log_lhd = sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2})

        #sympy.pprint(sympy.simplify(sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2, mu_s: 0, sigma_s: 1})))
        #sympy.pprint(sympy.simplify(sym_log_lhd.subs({mu_s: 0, sigma_s: 1}).diff(lamda_s)))
        #assert False

        loss_fn = ( sympy.diff(sym_log_lhd, mu_s)**2
                    + sympy.diff(sym_log_lhd, sigma_s)**2
                    + sympy.diff(sym_log_lhd, rho_s)**2
                    + sympy.diff(sym_log_lhd, lamda_s)**2 )
        #pv_loss_fn = sympy.log(loss_fn.subs({z1_s: pv_1, z2_s: pv_2})) \
        #    - 10*pv_sym_log_lhd

        pv_loss_fn = sympy.log(loss_fn.subs({z1_s: pv_1, z2_s: pv_2}))
        
        """
        loss_fn = ( sympy.diff(pv_sym_log_lhd, mu_s)**2
                    + sympy.diff(pv_sym_log_lhd, sigma_s)**2
                    + sympy.diff(pv_sym_log_lhd, rho_s)**2
                    + sympy.diff(pv_sym_log_lhd, lamda_s)**2 )
        pv_loss_fn = loss_fn #sympy.log(loss_fn.subs({z1_s: pv_1, z2_s: pv_2})) 
        sympy.pprint( loss_fn )
        assert False
        """
        #pv_loss_fn = -pv_sym_log_lhd
        theano_loss = theano_function(
            (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
            [pv_loss_fn,],
            dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})

        sym_gradients = []
        for sym in []: #(mu_s, sigma_s, rho_s, lamda_s):
            sym_grad = sympy.diff(loss_fn, sym)
            pv_sym_grad = sym_grad.subs({z1_s: pv_1, z2_s: pv_2})
            sym_gradients.append( pv_sym_grad )

        theano_gradient = None
        #theano_gradient = theano_function(
        #    (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
        #    sym_gradients,
        #    dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})
        with open("cached_fn.obj", "wb") as fp:
            pickle.dump((theano_loss, theano_gradient), fp)
    
        
    def calc_loss(theta, z1, z2):
        mu, sigma, rho, lamda = theta
        return theano_loss(
            numpy.repeat(mu, len(z1)),
            numpy.repeat(sigma, len(z1)),
            numpy.repeat(rho, len(z1)),
            numpy.repeat(lamda, len(z1)),
            z1, z2 ).sum()
    
    def calc_gradient(theta, z1, z2, fix_mu, fix_sigma):
        mu, sigma, rho, lamda = theta

        res = theano_gradient(
            numpy.repeat(mu, len(z1)),
            numpy.repeat(sigma, len(z1)),
            numpy.repeat(rho, len(z1)),
            numpy.repeat(lamda, len(z1)),
            z1, z2 )
        return numpy.array( [x.sum() for x in res] )
    
    rv = (calc_loss, calc_gradient)
    return rv


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
    
    return [(numpy.array(x.argsort().argsort(), dtype=float)+1)/(N+1) 
            for x in sim_values], sim_values

#params = [1, 1, 0.9, 0.5]
#(r1_r, r2_r), (r1_v, r2_v) = simulate_values(10, params)

#loss, grad = build_squared_gradient_loss()
#print loss(params, r1_v, r2_v)
#print grad(params, r1_v, r2_v, False, False)
#print grad



#a, b, c = build_copula_mixture_loss_and_grad_natural_params()
#with open("cached_gaussian_mix_cdf.obj", "w") as fp:
#    rv = pickle.dump(a, fp)                
