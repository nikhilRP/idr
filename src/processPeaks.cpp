/*****************************************************************************
  processPeaks.cpp

  (c) 2014 - Nikhil R Podduturi
  J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/

#include "processPeaks.h"
#include "bedtools/lineFileUtilities.h"
#include "bedtools/genomeFile.h"
#include "bedtools/bedFile.h"
#include "bedtools/interval.h"

/*
Constructor
*/
ProcessPeaks::ProcessPeaks( string peakFileA,
        string peakFileB,
        string genomeFile) {
    _peakFileA = peakFileA;
    _peakFileB = peakFileB;
    _genomeTable = genomeFile;

    _peakA = new BedFile(peakFileA);
    _peakB = new BedFile(peakFileB);
    _genome = new GenomeFile(genomeFile);

    FindOverlaps();
}


/*
Destructor
*/
ProcessPeaks::~ProcessPeaks(void) {
}


int compare_unsigned_int (const void *a, const void *b)
{
    unsigned int *a_i = (unsigned int *)a;
    unsigned int *b_i = (unsigned int *)b;
    if (*a_i < *b_i)
        return -1;
    else if (*a_i > *b_i)
        return 1;
    else
        return 0;
}


int compare_interval_by_start (const void *a, const void *b)
{
    struct interval *a_i = (struct interval *)a;
    struct interval *b_i = (struct interval *)b;
    if (a_i->start < b_i->start)
        return -1;
    else if (a_i->start > b_i->start)
        return 1;
    else
        return 0;
}


void parse_files(GenomeFile *_genome,
        map<string,CHRPOS> *_offsets,
        BedFile *_bedA,
        BedFile *_bedB,
        vector<struct interval> *_A,
        vector<struct interval> *_B)
{
    vector<string> chromList =  _genome->getChromList();
    sort( chromList.begin(), chromList.end() );

    CHRPOS curr_offset = 0;
    for (size_t c = 0; c < chromList.size(); ++c) {
        string currChrom = chromList[c];
        CHRPOS currChromSize = _genome->getChromSize(currChrom);
        (*_offsets)[currChrom] = curr_offset;
        curr_offset += currChromSize;
    }

    _bedA->loadBedFileIntoVector();
    _bedB->loadBedFileIntoVector();

    string last_chr;
    CHRPOS last_proj = 0;

    struct interval ivl;
    CHRPOS projected_start;
    CHRPOS projected_end;
    for (size_t i = 0; i < _bedA->bedList.size(); ++i) {
        if (_bedA->bedList[i].chrom.compare(last_chr) != 0)
            last_proj = (*_offsets)[_bedA->bedList[i].chrom];
        projected_start = last_proj + _bedA->bedList[i].start;
        projected_end = last_proj +  _bedA->bedList[i].end;
        ivl.start = projected_start;
        ivl.end   = projected_end - 1;
        _A->push_back(ivl);
    }
    fprintf(stderr, "  Peaks found in file 1 - %ld\n", _bedA->bedList.size());

    last_chr = "";
    for (size_t i = 0; i < _bedB->bedList.size(); ++i) {
        if (_bedB->bedList[i].chrom.compare(last_chr) != 0)
            last_proj = (*_offsets)[_bedB->bedList[i].chrom];
        projected_start = last_proj + _bedB->bedList[i].start;
        projected_end = last_proj +  _bedB->bedList[i].end;
        ivl.start = projected_start;
        ivl.end   = projected_end - 1;
        _B->push_back(ivl);
    }
    fprintf(stderr, "  Peaks found in file 2 - %ld\n", _bedB->bedList.size());
}

/*
 * Basic Binary search
 */
unsigned int bsearch_seq(unsigned int key,
        unsigned int *D,
        unsigned int D_size,
        int lo,
        int hi)
{
    int i = 0;
    unsigned int mid;
    while ( hi - lo > 1) {
        ++i;
        mid = (hi + lo) / 2;
        if ( D[mid] < key )
            lo = mid;
        else
            hi = mid;
    }
    return hi;
}


unsigned int per_interval_overlaps(struct interval *A,
        unsigned int size_A,
        struct interval *B,
        unsigned int size_B,
        unsigned int *R)
{
    unsigned int i, O = 0;

    unsigned int *B_starts =
        (unsigned int *) malloc(size_B * sizeof(unsigned int));
    unsigned int *B_ends =
        (unsigned int *) malloc(size_B * sizeof(unsigned int));

    for (i = 0; i < size_B; i++) {
        B_starts[i] = B[i].start;
        B_ends[i] = B[i].end;
    }

    qsort(B_starts, size_B, sizeof(unsigned int), compare_unsigned_int);
    qsort(B_ends, size_B, sizeof(unsigned int), compare_unsigned_int);

    for (i = 0; i < size_A; i++) {
        unsigned int num_cant_before = bsearch_seq(A[i].start, B_ends, size_B, -1, size_B);
        unsigned int b = bsearch_seq(A[i].end, B_starts, size_B, -1, size_B);
        while ( ( B_starts[b] == A[i].end) && b < size_B)
            ++b;
        unsigned int num_cant_after = size_B - b;
        unsigned int num_left = size_B - num_cant_before - num_cant_after;
        O += num_left;
        R[i] = num_left;
    }
    free(B_starts);
    free(B_ends);
    return O;
}


unsigned int get_intersections(struct interval *A,
        unsigned int size_A,
        struct interval *B,
        unsigned int size_B,
        unsigned int **R,
        unsigned int **E)
{
    *R = (unsigned int *) malloc(size_A * sizeof(unsigned int));

    unsigned int O = per_interval_overlaps(A,
            size_A,
            B,
            size_B,
            *R);
    int i;
    for (i = 1; i < size_A; i++)
        (*R)[i] = (*R)[i] + (*R)[i-1];

    for (i = 0; i < size_B; i++)
        B[i].order = i;

    qsort(B, size_B, sizeof(struct interval), compare_interval_by_start);

    unsigned int *B_starts =
        (unsigned int *) malloc(size_B * sizeof(unsigned int));
    for (i = 0; i < size_B; i++)
        B_starts[i] = B[i].start;

    *E = (unsigned int *) malloc(O * sizeof(unsigned int));

    unsigned int start = 0, end;
    for (i = 0; i < size_A; i++) {
        if (i != 0)
            start = (*R)[i - 1];
        end = (*R)[i];
        if (end - start > 0) {
            unsigned int from = bsearch_seq(A[i].end,
                    B_starts,
                    size_B,
                    -1,
                    size_B);

            while ( ( B_starts[from] == A[i].end) && from < size_B)
                ++from;

            while (  (end - start) > 0 ) {
                if ( (A[i].start <= B[from].end) &&
                        (A[i].end >= B[from].start) ) {
                    (*E)[start] = B[from].order;
                    start++;
                }
                --from;
            }
        }
    }
    free(B_starts);
    return O;
}


/*
 * primary method to find overlaps among the peak files
 */
void ProcessPeaks::FindOverlaps() {

    vector<struct interval> A, B;

    fprintf(stderr, "Processing peaks - started\n");
    parse_files(_genome, &_offsets, _peakA, _peakB, &A, &B);
    fprintf(stderr, "Processing peaks - Done\n");

    uint32_t tot_overlaps = get_intersections(&A[0],
            A.size(),
            &B[0],
            B.size(),
            &overlap_index_A,
            &overlap_index_B);

    fprintf( stderr, "Overlaps found among peak files - %d\n", tot_overlaps );
}
