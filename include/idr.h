/*************************************************************************************
 idr.h

 (c) 2014 - Nikhil R Podduturi
 J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

 Licensed under the GNU General Public License 2.0 license.
**************************************************************************************/
#ifndef IDR_H
#define IDR_H

#include <cmath>
#include <float.h>
#include <numeric>

using namespace std;

double RationalApproximation(double t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

double NormalCDFInverse(double p)
{
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

void calculate_quantiles(
    double rho,
    size_t n_samples,
    double* x_cdf, 
    double* y_cdf,
    double* updated_density
    )
{
    for(int i=0; i<n_samples; ++i)
    {
        double a = pow(NormalCDFInverse(x_cdf[i]), 2) 
                   + pow(NormalCDFInverse(y_cdf[i]), 2);
        double b = NormalCDFInverse(x_cdf[i]) * NormalCDFInverse(y_cdf[i]);
        updated_density[i] = exp( -log(1 - pow(rho, 2)) / 2 
                                  - rho/(2 * (1 - pow(rho, 2))) * (rho*a-2*b));
    }
}

double cost_function(
    double rho,
    size_t n_samples,
    double* x_cdf, 
    double* y_cdf,
    double* ez )
{

    double* new_density = (double*) calloc(n_samples, sizeof(double));
    calculate_quantiles(rho, 
                        n_samples,
                        x_cdf, 
                        y_cdf, 
                        new_density);

    double cop_den = 0.0;
    for(int i=0; i<n_samples; ++i)
    {
        cop_den = cop_den + (ez[i] * log(new_density[i]));
    }
    free(new_density);
    return -cop_den;
}

double maximum_likelihood(
    size_t n_samples,
    double* x_cdf,
    double* y_cdf,
    double* ez)
{
    double ax = -0.998;
    double bx = 0.998;
    double tol = 0.00001;

    /*  c is the squared inverse of the golden ratio */
    const double c = (3. - sqrt(5.)) * .5;

    /* Local variables */
    double a, b, d, e, p, q, r, u, v, w, x;
    double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

    /*  eps is approximately the square root of the relative machine precision. */
    eps = DBL_EPSILON;
    tol1 = eps + 1.;/* the smallest 1.000... > 1 */
    eps = sqrt(eps);

    a = ax;
    b = bx;
    v = a + c * (b - a);
    w = v;
    x = v;

    d = 0.;/* -Wall */
    e = 0.;
    fx = cost_function(x, n_samples, x_cdf, y_cdf, ez);
    fv = fx;
    fw = fx;
    tol3 = tol / 3.;

    for(;;)
    {
        xm = (a + b) * .5;
        tol1 = eps * fabs(x) + tol3;
        t2 = tol1 * 2.;

        /* check stopping criterion */
        if (fabs(x - xm) <= t2 - (b - a) * .5) break;
        p = 0.;
        q = 0.;
        r = 0.;
        if (fabs(e) > tol1)
        { /* fit parabola */
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.;
            if (q > 0.) p = -p; else q = -q;
            r = e;
            e = d;
        }

        if (fabs(p) >= fabs(q * .5 * r) ||
            p <= q * (a - x) || p >= q * (b - x))
        { /* a golden-section step */
            if (x < xm) e = b - x; else e = a - x;
            d = c * e;
        }
        else
        { /* a parabolic-interpolation step */
            d = p / q;
            u = x + d;
            /* f must not be evaluated too close to ax or bx */
            if (u - a < t2 || b - u < t2)
            {
                d = tol1;
                if (x >= xm) d = -d;
            }
        }

        /* f must not be evaluated too close to x */
        if (fabs(d) >= tol1)
            u = x + d;
        else if (d > 0.)
            u = x + tol1;
        else
            u = x - tol1;

        fu = cost_function(u, n_samples, x_cdf, y_cdf, ez);

        /*  update  a, b, v, w, and x */
        if (fu <= fx)
        {
            if (u < x) b = x; else a = x;
            v = w;    w = x;   x = u;
            fv = fw; fw = fx; fx = fu;
        }
        else
        {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x)
            {
                v = w; fv = fw;
                w = u; fw = fu;
            }
            else if (fu <= fv || v == x || v == w)
            {
                v = u; fv = fu;
            }
        }
    }
    return x;
}

double gaussian_loglikelihood(
    size_t n_samples,
    
    double*  x1_pdf, 
    double*  x2_pdf,
    double*  x1_cdf, 
    
    double*  y1_pdf, 
    double*  y2_pdf,
    double*  y1_cdf, 
    
    double p, double rho)
{
    double* density_c1 = (double*) calloc( sizeof(double), n_samples );
    double l0 = 0.0;

    calculate_quantiles(rho, n_samples, x1_cdf, y1_cdf, density_c1);
    for(int i=0; i<n_samples; ++i)
    {
        /* BUG XXX shouldnt line 2 be (1.0-p)*1.0*x2_pdf[i]*y2_pdf[i]) */
        l0 = l0 + log(p*density_c1[i]*x1_pdf[i]*y1_pdf[i] 
                      + (1.0-p)*1.0*x2_pdf[i]*y2_pdf[i]);
    }
    free(density_c1);
    return l0;
}

void estep_gaussian(
    size_t n_samples,
    double* x1_pdf, double* x2_pdf,
    double* x1_cdf, 
    double* y1_pdf, double* y2_pdf,
    double* y1_cdf, 
    double* ez, double p, double rho)

{
    /* update density_c1 */
    double* density_c1 = (double*) calloc( sizeof(double), n_samples );
    calculate_quantiles(rho, n_samples, x1_cdf, y1_cdf, density_c1);
    
    for(int i=0; i<n_samples; ++i)
    {
        /* XXX BUG? Shouldn't the last line be 
           ... +(1-p)*(1-density_c1[i])*x2_pdf[i]*y2_pdf[i]) */
        ez[i] = p * density_c1[i] * x1_pdf[i] * y1_pdf[i] 
            / (p * density_c1[i] * x1_pdf[i] * y1_pdf[i] 
               + (1-p) * 1 * x2_pdf[i] * y2_pdf[i]);
    }

    free(density_c1);
}

/*
 * Basic Binary search
 */
size_t bsearch(
    double key,
    double* D,
    size_t n)
{
    int lo = 0;
    int hi = n;
    unsigned int mid;
    while (hi - lo > 1) {
        mid = (hi + lo) / 2;
        if ( D[mid] < key )
            lo = mid;
        else
            hi = mid;
    }
    return hi;
}

/* use a histogram estimator to estimate the marginal distributions */
void estimate_marginals(
    size_t n_samples,
    float* input, 
    double* pdf_1, 
    double* pdf_2,
    double* cdf_1, 
    /* the estimated mixture paramater for each point */
    double* ez, 
    
    size_t nbins,
    
    /* the global mixture param */
    double p)
{
    const double bin_width = ((double)n_samples)/nbins;
    double* breaks = (double*) alloca(sizeof(double)*(nbins+1));
    /* make this a little smaller to avoid rounding errors */
    breaks[0] = 0. - 1e-6;
    for(int i=1; i<(nbins+1); ++i)
    {
        breaks[i] = i*bin_width;
    }
    /* to avoid rounding errors */
    breaks[nbins] += 1e-6;
    
    double* temp_cdf_1 = (double*) calloc(nbins, sizeof(double)); 
    double* temp_pdf_1 = (double*) calloc(nbins, sizeof(double)); 
    double* temp_pdf_2 = (double*) calloc(nbins, sizeof(double));
    
    /* estimate the weighted signal fraction and noise fraction sums */
    double sum_ez = 0;
    for(int i=0; i<n_samples; ++i)
        sum_ez += ez[i];
    double dup_sum_ez = n_samples - sum_ez;

    /* find which bin each value corresponds to, and 
       set this to the correct value */
    for(int i=0; i<n_samples; ++i)
    {
        double val = (double) bsearch(input[i], breaks, nbins);
        cdf_1[i] = val;
        pdf_1[i] = val;
        pdf_2[i] = val;
    }

    /* scale factor for the histogram estimator - I have no idea where this is 
       coming from or what the point is */
    const double scale = ((n_samples+nbins)/(bin_width*(n_samples+nbins+1.0)));
    /* for each bin, estimate the total probability
       mass from the items that fall into this bin */
    for(int k=0; k<nbins; ++k)
    {
        double sum_1 = 0.0;
        double sum_2 = 0.0;
        for(int m=0; m<n_samples; ++m)
        {
            if(cdf_1[m] == k+1)
            {
                sum_1 = sum_1 + ez[m];
                sum_2 = sum_2 + (1.0 - ez[m]);
            }
        }
        
        temp_pdf_1[k] = scale*(sum_1 + 1)/(sum_ez + nbins);
        temp_pdf_2[k] = scale*(sum_2 + 1)/(dup_sum_ez + nbins);

        for(int m=0; m<n_samples; ++m)
        {
            if(pdf_1[m] == k+1)
                pdf_1[m] = temp_pdf_1[k];
            if(pdf_2[m] == k+1)
                pdf_2[m] = temp_pdf_2[k];
        }

        temp_cdf_1[k] = temp_pdf_1[k] * bin_width;
    }

    double* new_cdf_1 = (double*) calloc(nbins, sizeof(double)); 

    new_cdf_1[0] = 0.0;

    // Naive sequential scan
    for(int p=1; p<nbins; ++p)
    {
        new_cdf_1[p] = temp_cdf_1[p-1] + new_cdf_1[p-1];
    }

    for(int l=0; l<n_samples; ++l)
    {
        int i = lroundf(cdf_1[l]);
        double b = input[l] - breaks[i-1];
        cdf_1[l] = new_cdf_1[i-1] + temp_pdf_1[i-1] * b;
    }

    free(temp_pdf_1);
    free(temp_pdf_2);
    free(temp_cdf_1);
    free(new_cdf_1);
}

void mstep_gaussian(
    double* p0, double* rho,
    size_t n_samples,
    double* x1_cdf, 
    double* y1_cdf,
    double* ez)
{
    *rho = maximum_likelihood(
        n_samples, x1_cdf, y1_cdf, ez);

    double sum_ez = 0;
    for(int i = 0; i < n_samples; i++)
    { sum_ez += ez[i]; }
    *p0 = sum_ez/(double)n_samples;
}


void em_gaussian(
    size_t n_samples,
    float* x, 
    float* y,
    double* localIDR)
{
    int i;
    
    double* ez = (double*) malloc( sizeof(double)*n_samples );
    int mid = round((float) n_samples/2);
    for(i = 0; i<n_samples/2; i++)
        ez[i] = 0.9;
    for(i = n_samples/2; i<n_samples; i++)
        ez[i] = 0.1;

    /* initialize the default configuration options */
    double p0 = 0.5;
    double rho = 0.0;
    double eps = 0.01;

    /* Initialize the set of break points for the histogram averaging */
    size_t n_bins = 50;
    float bin_width = (float)(n_samples-1)/n_bins;
    
    /*
     * CDF and PDF vectors for the input vectors.
     * Updated everytime for a EM iteration.
     */
    double* x1_pdf = (double*) calloc(sizeof(double), n_samples);
    double* x2_pdf = (double*) calloc(sizeof(double), n_samples);
    double* x1_cdf = (double*) calloc(sizeof(double), n_samples);
    double* y1_pdf = (double*) calloc(sizeof(double), n_samples);
    double* y2_pdf = (double*) calloc(sizeof(double), n_samples);
    double* y1_cdf = (double*) calloc(sizeof(double), n_samples);
    
    /* Likelihood vector */
    double likelihood[3] = {0,0,0};

    int iter_counter;
    for(iter_counter=0;;iter_counter++)
    {
        estimate_marginals(n_samples, x, 
                           x1_pdf, x2_pdf, x1_cdf, 
                           ez,
                           n_bins, 
                           p0);
        estimate_marginals(n_samples, y, 
                           y1_pdf, y2_pdf, y1_cdf, 
                           ez,
                           n_bins, 
                           p0);

        mstep_gaussian(&p0, &rho, n_samples, 
                       x1_cdf, y1_cdf, ez);

        estep_gaussian(n_samples,
                       x1_pdf, x2_pdf, 
                       x1_cdf, 
                       y1_pdf, y2_pdf, 
                       y1_cdf, 
                       ez, 
                       p0, rho);

        double l = gaussian_loglikelihood(
            n_samples,
            x1_pdf, x2_pdf, x1_cdf, 
            y1_pdf, y2_pdf, y1_cdf, 
            p0, rho);

        /* update the likelihood list */
        likelihood[0] = likelihood[1];
        likelihood[1] = likelihood[2];
        likelihood[2] = l;
        printf("%i\t%e\n", iter_counter, l);
        
        if (iter_counter > 1)
        {
            /* Aitken acceleration criterion checking for breaking the loop */
            double a_cri = likelihood[0] + (
                likelihood[1]-likelihood[0])
                / (1-(likelihood[2]-likelihood[1])/(
                       likelihood[1]-likelihood[0]));
            if ( abs(a_cri-likelihood[2]) <= eps )
            { break; }
        }

    }
    
    for(i=0; i<n_samples; ++i)
    {
        localIDR[i] = 1.0 - ez[i];
    }
    fprintf(stderr, "Finished running IDR on the datasets\n");
    fprintf(stderr, "Final P value = %.15g\n", p0);
    fprintf(stderr, "Final rho value = %.15g\n", rho);
    fprintf(stderr, "Total iterations of EM - %d\n", iter_counter-1);
    
    free(x1_pdf);
    free(x2_pdf);
    free(x1_cdf);
    free(y1_pdf);
    free(y2_pdf);
    free(y1_cdf);

}
#endif
