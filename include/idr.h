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
    vector<double>& x1_pdf, vector<double>& x2_pdf,
    vector<double>& x1_cdf, vector<double>& x2_cdf,
    vector<double>& y1_pdf, vector<double>& y2_pdf,
    vector<double>& y1_cdf, vector<double>& y2_cdf,
    double p, double rho)
{
    vector<double> density_c1( x1_pdf.size() );
    double l0 = 0.0;

    calculate_quantiles(rho, x1_cdf.size(), 
                        x1_cdf.data(), y1_cdf.data(), density_c1.data());
    for(int i=0; i<density_c1.size(); ++i)
    {
        l0 = l0 + log(p * density_c1[i] * x1_pdf[i] * y1_pdf[i] + (1.0 - p) * 1.0 * x2_pdf[i] * y2_pdf[i]);
    }
    return l0;
}

void estep_gaussian(
    size_t n_samples,
    double* x1_pdf, double* x2_pdf,
    double* x1_cdf, double* x2_cdf,
    double* y1_pdf, double* y2_pdf,
    double* y1_cdf, double* y2_cdf,
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

/* use a histogram estimator to estimate the marginal distributions */
void estimate_marginals(
    vector<float>& input, 
    vector<double>& breaks,
    vector<double>& pdf_1, 
    vector<double>& pdf_2,
    vector<double>& cdf_1, 
    vector<double>& cdf_2,
    /* the estimated mixture paramater for each point */
    vector<double>& ez, 
    double p)
{
    size_t n_samples = input.size();
    int nbins = breaks.size() - 1;

    double* temp_cdf_1 = (double*) calloc(nbins, sizeof(double)); 
    double* temp_cdf_2 = (double*) calloc(nbins, sizeof(double));
    double* temp_pdf_1 = (double*) calloc(nbins, sizeof(double)); 
    double* temp_pdf_2 = (double*) calloc(nbins, sizeof(double));
    
    for(int i=0; i<n_samples; ++i)
    {
        std::vector<double>::iterator low = lower_bound(
            breaks.begin(), breaks.end(), (double)input[i]);
        cdf_1[i] = low - breaks.begin();
    }

    int first_size = round((double)(n_samples*p));
    double bin_width = breaks[1] - breaks[0];
    
    cdf_2 = cdf_1;
    pdf_1 = cdf_1;
    pdf_2 = cdf_1;

    /* estimate the weighted signal fraction and noise fraction sums */
    double sum_ez = 0;
    for(int i=0; i<n_samples; ++i)
        sum_ez += ez[i];
    double dup_sum_ez = n_samples - sum_ez;

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

        temp_pdf_1[k] = (sum_1 + 1) / (sum_ez + nbins) / bin_width * (n_samples + 50) / (n_samples + 51);
        temp_pdf_2[k] = (sum_2 + 1) / (dup_sum_ez + nbins) / bin_width * (n_samples + 50) / (n_samples  + 51);

        for(int m=0; m<n_samples; ++m)
        {
            if(pdf_1[m] == k+1)
                pdf_1[m] = temp_pdf_1[k];
            if(pdf_2[m] == k+1)
                pdf_2[m] = temp_pdf_2[k];
        }

        temp_cdf_1[k] = temp_pdf_1[k] * bin_width;
        temp_cdf_2[k] = temp_pdf_2[k] * bin_width;
    }

    double* new_cdf_1 = (double*) calloc(nbins, sizeof(double)); 
    double* new_cdf_2 = (double*) calloc(nbins, sizeof(double)); 

    new_cdf_1[0] = 0.0;
    new_cdf_2[0] = 0.0;

    // Naive sequential scan
    for(int p=1; p<nbins; ++p)
    {
        new_cdf_1[p] = temp_cdf_1[p-1] + new_cdf_1[p-1];
        new_cdf_2[p] = temp_cdf_2[p-1] + new_cdf_2[p-1];
    }

    for(int l=0; l<n_samples; ++l)
    {
        int i = lroundf(cdf_1[l]);
        double b = input[l] - breaks[i-1];
        cdf_1[l] = new_cdf_1[i-1] + temp_pdf_1[i-1] * b;
        cdf_2[l] = new_cdf_2[i-1] + temp_pdf_2[i-1] * b;
    }

    free(temp_pdf_1);
    free(temp_pdf_2);
    free(temp_cdf_1);
    free(temp_cdf_2);
    free(new_cdf_1);
    free(new_cdf_2);
}

void mstep_gaussian(
    vector<float>& x, vector<float> y,
    vector<double>& breaks, double* p0, double* rho,
    vector<double>& x1_pdf, vector<double>& x2_pdf,
    vector<double>& x1_cdf, vector<double>& x2_cdf,
    vector<double>& y1_pdf, vector<double>& y2_pdf,
    vector<double>& y1_cdf, vector<double>& y2_cdf,
    vector<double>& ez)
{
    estimate_marginals(x, breaks, 
                       x1_pdf, x2_pdf, 
                       x1_cdf, x2_cdf, 
                       ez, *p0);
    estimate_marginals(y, breaks, 
                       y1_pdf, y2_pdf, 
                       y1_cdf, y2_cdf, 
                       ez, *p0);

    *rho = maximum_likelihood(
        x1_cdf.size(), x1_cdf.data(), y1_cdf.data(), ez.data());

    double sum_ez = accumulate(ez.begin(), ez.end(), 0.0);
    *p0 = sum_ez/(double)x.size();
}


void em_gaussian(
    vector<float>& x, vector<float>& y,
    vector< pair<int, double> >& idrLocal)
{
    vector<double> ez( x.size() );

    int mid = round((float) x.size()/2);

    fill(ez.begin(), ez.begin()+mid, 0.9);
    fill(ez.begin()+mid, ez.end(), 0.1);

    float bin_width = (float)(x.size()-1)/50;

    double p0 = 0.5;
    double rho = 0.0;
    double eps = 0.01;

    /* Breaks for binning the data */
    vector<double> breaks(51);
    for(int i=0; i<breaks.size(); ++i)
    {
        if (i == 0)
        {
            breaks[i] = (double)1-bin_width/100;
        }
        else
        {
            breaks[i] = breaks[i-1] + (double)(x.size()-1+bin_width/50)/50;
        }
    }

    /*
     * CDF and PDF vectors for the input vectors.
     * Updated everytime for a EM iteration.
     */
    vector<double> x1_pdf(x.size()), x2_pdf(x.size()), x1_cdf(x.size()), x2_cdf(x.size());
    vector<double> y1_pdf(x.size()), y2_pdf(x.size()), y1_cdf(x.size()), y2_cdf(x.size());

    fprintf(stderr, "    Initialising the marginals\n");

    mstep_gaussian(x, y, breaks, &p0, &rho, x1_pdf, x2_pdf,
        x1_cdf, x2_cdf, y1_pdf, y2_pdf, y1_cdf, y2_cdf, ez);

    /* Likelihood vector */
    vector<double> likelihood;
    double li = gaussian_loglikelihood(x1_pdf, x2_pdf, x1_cdf, x2_cdf,
        y1_pdf, y2_pdf, y1_cdf, y2_cdf, p0, rho);

    likelihood.push_back(li);
    fprintf(stderr, "    Done\n");

    bool flag = true;
    int i = 1;

    /* can do better. Jus replicating IDR R style coding */
    int iter_counter = 1;
    while(flag)
    {
        estep_gaussian(x1_pdf.size(),
                       x1_pdf.data(), x2_pdf.data(), 
                       x1_cdf.data(), x2_cdf.data(),
                       y1_pdf.data(), y2_pdf.data(), 
                       y1_cdf.data(), y2_cdf.data(), 
                       ez.data(), p0, rho);

        mstep_gaussian(x, y, breaks, &p0, &rho, x1_pdf, x2_pdf,
            x1_cdf, x2_cdf, y1_pdf, y2_pdf, y1_cdf, y2_cdf, ez);

        double l = gaussian_loglikelihood(x1_pdf, x2_pdf, x1_cdf, x2_cdf,
            y1_pdf, y2_pdf, y1_cdf, y2_cdf, p0, rho);
        likelihood.push_back(l);
        printf("%i\t%e\n", iter_counter, l);
        
        if (i > 1)
        {
            /* Aitken acceleration criterion checking for breaking the loop */
            double a_cri = likelihood[i-2] + (likelihood[i-1] - likelihood[i-2])
                / (1-(likelihood[i]-likelihood[i-1])/(likelihood[i-1]-likelihood[i-2]));
            if ( std::abs(a_cri-likelihood[i]) <= eps )
            {
                flag = false;
            }
        }
        i++;
        iter_counter++;
    }
    vector<double> temp(ez.size());
    for(int i=0; i<ez.size(); ++i)
    {
        double a = 1.0;
        idrLocal[i].first = i+1;
        idrLocal[i].second = a-ez[i];
    }
    fprintf(stderr, "Finished running IDR on the datasets\n");
    fprintf(stderr, "Final P value = %.15g\n", p0);
    fprintf(stderr, "Final rho value = %.15g\n", rho);
    fprintf(stderr, "Total iterations of EM - %d\n", iter_counter-1);
}
#endif
