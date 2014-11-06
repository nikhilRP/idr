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

void em_gaussian(
    size_t n_samples,
    float* x, 
    float* y,
    double* localIDR);

#endif
