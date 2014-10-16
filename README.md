idr
===
[![Build Status](https://travis-ci.org/nikhilRP/idr.svg?branch=master)](https://travis-ci.org/nikhilRP/idr)

Irreproducible Discovery Rate for high throughput ChiP-seq experiments

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

The main contributors of IDR-GPU code:

  * Nikhil R Podduturi  - nikhilrp@stanford.edu
  * J. Seth Strattan    - jseth@stanford.edu
  * Anshul Kundaje      - akundaje@stanford.edu
  
J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine
