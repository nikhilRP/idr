/*****************************************************************************
  idr.cpp

  (c) 2014 - Nikhil R Podduturi
  J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/
#include <stdio.h>
#include <string>
#include <stdlib.h>

#include <processPeaks.h>
#include <ranker.h>
#include <idr.h>

using namespace std;

#define PROGRAM_NAME "IDR"
#define PACKAGE_VERSION "0.1"

#define PARAMETER_CHECK(param, paramLen, actualLen) (\
        strncmp(argv[i], param, min(actualLen, paramLen))== 0) \
    && (actualLen == paramLen)

void ShowHelp(void);

// Utility struct to keep track of final results
struct overlap
{
    string chr;
    CHRPOS start1;
    CHRPOS end1;
    double rankingMeasure1;
    CHRPOS start2;
    CHRPOS end2;
    double rankingMeasure2;
    double idrLocal;
    double idr;
};

struct sort_pred
{
    bool operator()(const std::pair<int, double> &left,
                    const std::pair<int, double> &right)
    {
        return left.second < right.second;
    }
};

#define DEFAULT_IDR_CUTOFF 0.025
#define DEFAULT_OFNAME "idrValues.txt"
#define DEFAULT_RANKING_MEASURE "signalValue"

void ShowHelp(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "Program: IDR (Irreproducible Discovery Rate)\n");
    fprintf(stderr, "Version: %s\n", PACKAGE_VERSION);
    fprintf(stderr, "Contact: Nikhil R Podduturi <nikhilrp@stanford.edu>\n\n");
    fprintf(stderr, "Usage:   idr [options] -a <bed> -b <bed> -g <bed>\n\n");
    fprintf(stderr, "Options:\n\n");
    fprintf(stderr,
        "   -o          Output filename (default: idrValues.txt)\n\n");
    fprintf(stderr,
        "   -idr        IDR cutoff (default: %e)\n\n", DEFAULT_IDR_CUTOFF);
    fprintf(stderr,
        "   -rank       Type of ranking measure (default: %s)\n\n",
        DEFAULT_RANKING_MEASURE);
    exit(1);
}

class Args {
    public:
    string bedAFile;
    string bedBFile;
    string genomeFile;

    string ofname;
    string rankingMeasure;
    float idrCutoff;

    Args( int argc, char* argv[] )
    {
        this->ofname = DEFAULT_OFNAME;
        this->rankingMeasure = DEFAULT_RANKING_MEASURE;
        this->idrCutoff = DEFAULT_IDR_CUTOFF;

        bool haveBedA = false;
        bool haveBedB = false;
        bool haveGenome = false;
        bool haveOutput = false;

        if(argc <= 1) ShowHelp();

        for(int i = 1; i < argc; i++) {
            int parameterLength = (int)strlen(argv[i]);
            if((PARAMETER_CHECK("-h", 2, parameterLength)) ||
               (PARAMETER_CHECK("--help", 6, parameterLength))) {
                ShowHelp();
            }
        }

        for(int i = 1; i < argc; i++) {
            int parameterLength = (int)strlen(argv[i]);
            if(PARAMETER_CHECK("-a", 2, parameterLength)) {
                if ((i+1) < argc) {
                    haveBedA = true;
                    this->bedAFile = argv[i + 1];
                    i++;
                }
            }
            else if(PARAMETER_CHECK("-b", 2, parameterLength)) {
                if ((i+1) < argc) {
                    haveBedB = true;
                    this->bedBFile = argv[i + 1];
                    i++;
                }
            }
            else if(PARAMETER_CHECK("-o", 2, parameterLength)) {
                if((i+1) < argc) {
                    haveOutput = true;
                    this->ofname = argv[i + 1];
                    i++;
                }
            }
            else if(PARAMETER_CHECK("-g", 2, parameterLength)) {
                if ((i+1) < argc) {
                    haveGenome = true;
                    this->genomeFile = argv[i + 1];
                    i++;
                }
            }
            else if(PARAMETER_CHECK("-rank", 5, parameterLength)) {
                if((i+1) < argc) {
                    this->rankingMeasure = argv[i + 1];
                    i++;
                }
            }
            else if(PARAMETER_CHECK("-idr", 4, parameterLength)) {
                if((i+1) < argc) {
                    this->idrCutoff = atof(argv[i + 1]);
                    i++;
                }
            }
            else {
                fprintf(stderr,
                   "*****ERROR: Unrecognized parameter: %s *****\n\n", argv[i]);
                ShowHelp();
            }
        }
        // make sure we have both input files
        if (!haveBedA || !haveBedB || !haveGenome) {
            fprintf(stderr, "*****ERROR: Need -a, -b and  -g files. *****");
            ShowHelp();
        }
    };
};

int
build_ranks_vector( ProcessPeaks *bc,
                    string rankingMeasure,
                    vector<overlap> &overlaps,
                    vector<float> &ranks_A,
                    vector<float> &ranks_B)
{
    vector<double> merge_A, merge_B, unmatched_merge_A, unmatched_merge_B;
    vector<unsigned int> tracker;

    unsigned int start = 0;
    for (size_t i = 0; i < bc->_peakA->bedList.size(); ++i) {
        if (rankingMeasure == "signalValue") {
            merge_A.push_back(bc->_peakA->bedList[i].signalValue);
        }
        else if(rankingMeasure == "pValue") {
            merge_A.push_back(bc->_peakA->bedList[i].pValue);
        }
        else if(rankingMeasure == "qValue") {
            merge_A.push_back(bc->_peakA->bedList[i].qValue);
        }
        merge_B.push_back(0);
        while (start < bc->overlap_index_A[i]) {
            tracker.push_back( bc->overlap_index_B[start] );
            double bSigVal = 0.0;
            if (rankingMeasure == "signalValue") {
                bSigVal = bc->_peakB->bedList[
                    bc->overlap_index_B[start] ].signalValue;
            }
            else if (rankingMeasure == "pValue") {
                bSigVal = bc->_peakB->bedList[
                    bc->overlap_index_B[start] ].pValue;
            }
            else if (rankingMeasure == "qValue") {
                bSigVal = bc->_peakB->bedList[
                    bc->overlap_index_B[start] ].qValue;
            }

            overlap o;
            string chr(bc->_peakA->bedList[i].chrom);
            o.chr = chr;
            o.start1 = bc->_peakA->bedList[i].start;
            o.end1 = bc->_peakA->bedList[i].end;
            o.rankingMeasure1 = merge_A[i];
            o.start2 = bc->_peakB->bedList[ bc->overlap_index_B[start] ].start;
            o.end2 = bc->_peakB->bedList[ bc->overlap_index_B[start] ].end;

            if (merge_B[i] == 0) {
                merge_B[i] = bSigVal;
                o.rankingMeasure2 = merge_B[i];
                overlaps.push_back(o);
            }
            else {
                merge_B[i] = double (merge_B[i] + bSigVal)/2.0;
                o.rankingMeasure2 = merge_B[i];
                overlaps.pop_back();
                overlaps.push_back(o);
            }
            ++start;
        }
    }
    sort(tracker.begin(), tracker.end());
    for (unsigned int i = 0; i < bc->_peakB->bedList.size(); ++i) {
        if(find(tracker.begin(), tracker.end(), i) == tracker.end()) {
            double bSigVal = 0.0;
            if (rankingMeasure == "signalValue") {
                bSigVal = bc->_peakB->bedList[i].signalValue;
            }
            else if (rankingMeasure == "pValue") {
                bSigVal = bc->_peakB->bedList[i].pValue;
            }
            else if (rankingMeasure == "qValue") {
                bSigVal = bc->_peakB->bedList[i].qValue;
            }
            merge_A.push_back( 0 );
            merge_B.push_back( bSigVal );
        }
    }

    for (unsigned int i=0; i < merge_A.size(); ++i) {
        if ( merge_A[i] != 0 && merge_B[i] != 0 ) {
            unmatched_merge_A.push_back( -merge_A[i] );
            unmatched_merge_B.push_back( -merge_B[i] );
        }
    }

    fprintf(stderr, "Number of overlaps after removing duplicates - %lu\n",
            unmatched_merge_A.size());

    fprintf(stderr, "Ranking overlaps\n");

    rank_vec(unmatched_merge_A, ranks_A, "average");
    rank_vec(unmatched_merge_B, ranks_B, "average");

    fprintf(stderr, "   Done\n");
    return 0;
}

int main(int argc, char* argv[])
{
    Args args = Args(argc, argv);
    ProcessPeaks *bc = new ProcessPeaks(
        args.bedAFile, args.bedBFile, args.genomeFile);

    vector<overlap> overlaps;

    /* Find the joint peak list, and rank them */
    vector<float> ranks_A, ranks_B;
    build_ranks_vector(
        bc, args.rankingMeasure,
        overlaps,
        ranks_A, ranks_B );

    vector< pair<int, double> > idr(ranks_A.size());

    fprintf(stderr, "Fit 2-component model - started\n");

    em_gaussian(ranks_A, ranks_B, idr);

    /* Make a copy of the local IDR vector, and then
       update the 'second' field to store the global
       IDR value (XXX use decent variable names...) */
    vector<double> idrLocal(ranks_A.size());

    for(int i=0; i<idr.size(); ++i)
    {
        idrLocal[i] = idr[i].second;
    }

    sort(idr.begin(), idr.end(), sort_pred());
    for(int i=1; i<idr.size(); ++i)
    {
        idr[i].second = idr[i].second + idr[i-1].second;
    }
    int num_peaks_passing_threshold = 0;
    for(int j=0; j<ranks_A.size(); ++j)
    {
        idr[j].second = idr[j].second/((double)j);
        if(idr[j].second <= args.idrCutoff) {
            num_peaks_passing_threshold += 1;
        }
    };
    sort(idr.begin(), idr.end());
    for (unsigned int i=0; i<idr.size(); ++i)
    {
        overlaps[i].idrLocal = idrLocal[i];
        overlaps[i].idr = idr[i].second;
    }
    std::filebuf fb;
    fb.open(args.ofname.c_str(), std::ios::out);
    std::ostream fout(&fb);
    fout.precision(15);
    for (unsigned int i=0; i < idr.size(); ++i)
    {
        if (overlaps[i].idr <= args.idrCutoff)
        {
            fout<<overlaps[i].chr<<"\t"<<
                overlaps[i].start1<<"\t"<<
                overlaps[i].end1<<"\t"<<
                overlaps[i].rankingMeasure1<<"\t"<<
                overlaps[i].start2<<"\t"<<
                overlaps[i].end2<<"\t"<<
                overlaps[i].rankingMeasure2<<"\t"<<
                std::fixed<<overlaps[i].idrLocal<<"\t"<<
                std::fixed<<overlaps[i].idr<<endl;
        }
    }
    fb.close();
    fprintf(stderr, "Number of peaks passing IDR cutoff of %f - %d\n",
            args.idrCutoff, num_peaks_passing_threshold);
    return 0;
}
