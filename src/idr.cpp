/*****************************************************************************
  idr.cpp

  (c) 2014 - Nikhil R Podduturi
  J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/
#include <stdio.h>
#include <string>
#include <stdlib.h>

#include "processPeaks.h"
#include "ranker.h"

using namespace std;

#define PROGRAM_NAME "IDR"
#define PACKAGE_VERSION "0.1"

#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)

void ShowHelp(void);

// Utility struct to keep track of final results
struct overlap {
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

int main(int argc, char* argv[])
{
	bool showHelp = false;

	string bedAFile;
	string bedBFile;
	string genomeFile;
	string rankingMeasure = "signalValue";

	bool haveBedA = false;
	bool haveBedB = false;
	bool haveGenome = false;

	if(argc <= 1) showHelp = true;

	for(int i = 1; i < argc; i++) {
		int parameterLength = (int)strlen(argv[i]);

		if((PARAMETER_CHECK("-h", 2, parameterLength)) ||
		(PARAMETER_CHECK("--help", 5, parameterLength))) {
			showHelp = true;
		}
	}

	if(showHelp) ShowHelp();

	for(int i = 1; i < argc; i++) {

		int parameterLength = (int)strlen(argv[i]);
		if(PARAMETER_CHECK("-a", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveBedA = true;
				bedAFile = argv[i + 1];
				i++;
			}
		}
		else if(PARAMETER_CHECK("-b", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveBedB = true;
				bedBFile = argv[i + 1];
				i++;
			}
		}
		else if(PARAMETER_CHECK("-g", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveGenome = true;
				genomeFile = argv[i + 1];
				i++;
			}
		}
		else if(PARAMETER_CHECK("-rank", 5, parameterLength)) {
			if((i+1) < argc) {
				rankingMeasure = argv[i + 1];
				i++;
			}
		}
		else {
			fprintf(stderr, "*****ERROR: Unrecognized parameter: %s *****\n\n", argv[i]);
			showHelp = true;
		}
	}
	// make sure we have both input files
	if (!haveBedA || !haveBedB || !haveGenome) {
		fprintf(stderr, "*****ERROR: Need -a and -b files. *****");
		showHelp = true;
	}

	if (!showHelp) {
		ProcessPeaks *bc = new ProcessPeaks(bedAFile, bedBFile, genomeFile);
		vector<double> merge_A, merge_B, unmatched_merge_A, unmatched_merge_B;
		vector<unsigned int> tracker;
		vector<overlap> overlaps;

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
					bSigVal = bc->_peakB->bedList[ bc->overlap_index_B[start] ].signalValue;
				}
				else if (rankingMeasure == "pValue") {
					bSigVal = bc->_peakB->bedList[ bc->overlap_index_B[start] ].pValue;
				}
				else if (rankingMeasure == "qValue") {
					bSigVal = bc->_peakB->bedList[ bc->overlap_index_B[start] ].qValue;
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

		fprintf(stderr, "Number of overlaps after removing duplicates - %lu\n", unmatched_merge_A.size());

		vector<int> ranks_A, ranks_B;
		rank_vec(unmatched_merge_A, ranks_A, "default");
		rank_vec(unmatched_merge_B, ranks_B, "default");

		return 0;
	}
	else {
		ShowHelp();
	}
	return 0;
}

void ShowHelp(void) {
	fprintf(stderr, "\n");
	fprintf(stderr, "Program: IDR (Irreproducible Discovery Rate)\n");
	fprintf(stderr, "Version: %s\n", PACKAGE_VERSION);
	fprintf(stderr, "Contact: Nikhil R Podduturi <nikhilrp@stanford.edu>\n\n");
	fprintf(stderr, "Usage:   idr [options] -a <bed> -b <bed> -g <bed>\n\n");
	fprintf(stderr, "Options:\n\n");
	fprintf(stderr, "	-idr		IDR cutoff (default: 0.025)\n\n");
	fprintf(stderr, "	-rank		Type of ranking measure (default: signal.value)\n\n");
    exit(1);
}
