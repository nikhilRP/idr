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

template <typename V>
void calculate_quantiles(V& x_cdf, V& y_cdf, V rho, V& density)
{
    for(int i=0; i<x_cdf.size(); ++i)
    {
        double a = pow(normcdfinv(x_cdf[i]), 2) + pow(normcdfinv(y_cdf[i]), 2);
        double b = normcdfinv(x_cdf[i]) * normcdfinv(y_cdf[i]);
        density[i] = exp( -log(1 - pow(rho, 2)) / 2 - rho / (2 * (1 - pow(rho, 2))) * (rho*a-2*b) );
    }
}

template <typename V>
double cost_function(V& x_cdf, V& y_cdf, V& ez, V rho)
{
    typedef typename V::value_type ValueType;

    vector<ValueType> density(x_cdf.size());
    vector<ValueType> new_density(x_cdf.size());

    calculate_quantiles(x_cdf, y_cdf, rho, density);

    ValueType cop_den = 0.0;
    for(int i=0; i<density.size(); ++i)
    {
        cop_den = cop_den + (ez[i] * log(density[i]));
    }
    return -cop_den;
}

void calculate_quantiles()
{

}

void sum_likelihood()
{

}

void get_ez()
{

}

template <typename V>
double maximum_likelihood(V& x_cdf, V& y_cdf, V& ez)
{
    typedef typename V::value_type ValueType;

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
    fx = cost_function(x_cdf, y_cdf, ez, x);
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

        fu = cost_function(x_cdf, y_cdf, ez, u);

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

template <typename V>
double gaussian_loglikelihood(V& x1_pdf, V& x2_pdf, V& x1_cdf, V& x2_cdf,
        V& y1_pdf, V& y2_pdf, V& y1_cdf, V& y2_cdf, double p, double rho)
{
    typedef typename V::value_type ValueType;

    vector<ValueType> density_c1( x1_pdf.size() );
    vector<ValueType> likelihood_vec( x1_pdf.size() );

    // rewrite this
    calculate_quantiles();
    sum_likelihood();

    double l0 = accumulate(likelihood_vec.begin(), likelihood_vec.end(), 0.0);

    density_c1.clear();
    density_c1.shrink_to_fit();

    likelihood_vec.clear();
    likelihood_vec.shrink_to_fit();

    return l0;
}

template <typename V>
void estep_gaussian(V& x1_pdf, V& x2_pdf, V& x1_cdf, V& x2_cdf,
        V& y1_pdf, V& y2_pdf, V& y1_cdf, V& y2_cdf, V& ez, double p, double rho)
{
    typedef typename V::value_type ValueType;

    vector<ValueType> density_c1( x1_pdf.size() );

    // Rewrite those GPU kernals
    calculate_quantiles();
    get_ez();

    density_c1.clear();
    density_c1.shrink_to_fit();
}

void nikhil(){

}

template <typename V, typename T, typename P, typename R>
void mstep_gaussian(V& x, V& y, T& breaks, P *p0,  R *rho,
    T& x1_pdf, T& x2_pdf, T& x1_cdf, T& x2_cdf, T& y1_pdf,
    T& y2_pdf, T& y1_cdf, T& y2_cdf, T& ez)
{
    estimate_marginals(x, breaks, x1_pdf, x2_pdf, x1_cdf, x2_cdf, ez, *p0);
    estimate_marginals(y, breaks, y1_pdf, y2_pdf, y1_cdf, y2_cdf, ez, *p0);

    *rho = maximum_likelihood(x1_cdf, y1_cdf, ez);

    double sum_ez = accumulate(ez.begin(), ez.end(), 0.0) / x.size();
    *p0 = sum_ez/(double)x.size();
}

template <typename V, typename T>
void em_gaussian(V& x, V& y, T& idrLocal)
{
    typedef typename V::value_type ValueType;

    vector<ValueType> ez( x.size() );
    vector<ValueType> breaks(51);

    int mid = round((float) x.size()/2);

    fill(ez.begin(), ez.begin()+mid, 0.9);
    fill(ez.begin()+mid, ez.end(), 0.1);

    float bin_width = (float)(x.size()-1)/50;

    // Constants chosen by Li and Anshul
    double p0 = 0.5;
    double rho = 0.0;
    double eps = 0.01;

    vector<double> likelihood;
    breaks[0] = (double)1-bin_width/100;
    iota(breaks.begin(), breaks.end(), (double)(x.size()-1+bin_width/50)/50);

    vector<double> x1_pdf, x2_pdf, x1_cdf, x2_cdf;
    vector<double> y1_pdf, y2_pdf, y1_cdf, y2_cdf;

    mstep_gaussian(x, y, breaks, &p0, &rho, x1_pdf, x2_pdf,
        x1_cdf, x2_cdf, y1_pdf, y2_pdf, y1_cdf, y2_cdf, ez);

    double li = gaussian_loglikelihood(x1_pdf, x2_pdf, x1_cdf, x2_cdf,
            y1_pdf, y2_pdf, y1_cdf, y2_cdf, p0, rho);

    likelihood.push_back(li);

    bool flag = true;
    int i = 1;

    //can do better. Jus replicating IDR R style coding
    int iter_counter = 1;
    while(flag)
    {
        estep_gaussian(x1_pdf, x2_pdf, x1_cdf, x2_cdf,
            y1_pdf, y2_pdf, y1_cdf, y2_cdf, ez, p0, rho);

        mstep_gaussian(x, y, breaks, &p0, &rho, x1_pdf, x2_pdf,
            x1_cdf, x2_cdf, y1_pdf, y2_pdf, y1_cdf, y2_cdf, ez);

        double l = gaussian_loglikelihood(x1_pdf, x2_pdf, x1_cdf, x2_cdf,
            y1_pdf, y2_pdf, y1_cdf, y2_cdf, p0, rho);
        likelihood.push_back(l);

        if (i > 1)
        {
            double a_cri = likelihood[i-2] + (likelihood[i-1] - likelihood[i-2])/(1-(likelihood[i]-likelihood[i-1])/(likelihood[i-1]-likelihood[i-2]));
            if ( std::abs(a_cri-likelihood[i]) <= eps )
            {
                flag = false;
            }
        }
        i++;
        iter_counter++;
    }
}

#endif

