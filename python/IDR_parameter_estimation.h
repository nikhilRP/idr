/*************************************************************************************
 idr.h

 (c) 2014 - Nikhil R Podduturi
 J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

 Licensed under the GNU General Public License 2.0 license.
**************************************************************************************/
#ifndef IDR_H
#define IDR_H

#include <math.h>
#include <float.h>
#include <assert.h>
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
    int n_samples,
    int* x, 
    int* y,
    double* IDRs,
    int print_status_msgs);

#endif
