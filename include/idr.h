/*************************************************************************************
 idr.h

 (c) 2014 - Nikhil R Podduturi
 J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

 Licensed under the GNU General Public License 2.0 license.
**************************************************************************************/
#ifndef IDR_H
#define IDR_H

#include <numeric>

#include <cmath>
#include <float.h>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

void em_gaussian(
    size_t n_samples,
    float* x, 
    float* y,
    double* localIDR);

#endif
