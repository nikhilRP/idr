/*************************************************************************************
 idr.h

 (c) 2014 - Nikhil R Podduturi
 J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

 Licensed under the GNU General Public License 2.0 license.
**************************************************************************************/
#ifndef IDR_H
#define IDR_H

#include <numeric>
using namespace std;

#include <cmath>
#include <float.h>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>

struct OptimizationRV
{
    int n_iters;
    double rho;
    double p;
};

struct OptimizationRV
em_gaussian(
    size_t n_samples,
    double* x, 
    double* y,
    double* localIDR);

#endif
