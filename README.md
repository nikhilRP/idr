Irreproducible Discovery Rate (IDR)
===
[![Build Status](https://travis-ci.org/nikhilRP/idr.svg?branch=master)](https://travis-ci.org/nikhilRP/idr)

<p align="justify">The IDR (Irreproducible Discovery Rate) framework is a uniﬁed approach to measure the reproducibility of ﬁndings identiﬁed from replicate experiments and provide highly stable thresholds based on reproducibility. Unlike the usual scalar measures of reproducibility, the IDR approach creates a curve, which quantitatively assesses when the ﬁndings are no longer consistent across replicates. In layman's terms, the IDR method compares a pair of ranked lists of identifications (such as ChIP-seq peaks). These ranked lists should not be pre-thresholded i.e. they should provide identifications across the entire spectrum of high confidence/enrichment (signal) and low confidence/enrichment (noise). The IDR method then fits the bivariate rank distributions over the replicates in order to separate signal from noise based on a defined confidence of rank consistency and reproducibility of identifications i.e the IDR threshold.</p>

<p align="justify">The method was developed by <a href="http://www.personal.psu.edu/users/q/u/qul12/index.html">Qunhua Li</a> and <a href="http://www.stat.berkeley.edu/~bickel/">Peter Bickel</a>'s group and is extensively used by the ENCODE and modENCODE  projects and is part of their ChIP-seq guidelines and standards.</p>

### Building IDR

* Get the current repo
```
git clone --recursive https://github.com/nikhilRP/idr.git
```
* Create a directory parallel to the repo (eg: build)
```
mkdir build
```
* Then follow the commands below 
```
cd build
cmake ../idr
make
```

### Usage

Assuming that your current working directory is the one you created. 

Eg: `build` directory from above

List all the options
 
```
./idr
```

Sample idr run using test peak files in the repo

```
./idr -a ../idr/test/data/peak1 -b ../idr/test/data/peak2 -g ../idr/test/data/genome_table.txt
```

### Usage

[bedtools2](https://github.com/arq5x/bedtools2) is used to parse the peak files 

The main contributors of IDR-GPU code:

  * Nikhil R Podduturi  - nikhilrp@stanford.edu
  * J. Seth Strattan    - jseth@stanford.edu
  * Anshul Kundaje      - akundaje@stanford.edu
  
J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine
