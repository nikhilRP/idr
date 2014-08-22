/*****************************************************************************
  processPeaks.h

  (c) 2014 - Nikhil R Podduturi
  J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/

#ifndef PEAK_PROCESS_H
#define PEAK_PROCESS_H

#include <bedtools/bedFile.h>
#include <bedtools/genomeFile.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

class ProcessPeaks {

public:
    // constructor
    // destructor
    ProcessPeaks(string peakFileA,
            string peakFileB,
            string genomeFile);
    ~ProcessPeaks(void);

    // data structures used by Peaks
    BedFile *_peakA, *_peakB;
    unsigned int *overlap_index_A;
    unsigned int *overlap_index_B;

private:

    string _peakFileA;
    string _peakFileB;
    string _genomeTable;
    GenomeFile *_genome;

    map<string,CHRPOS> _offsets;

    void FindOverlaps(void);
};

#endif
