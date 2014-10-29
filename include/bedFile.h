#ifndef BEDFILE_H
#define BEDFILE_H
#include "lineFileUtilities.h"
#include "interval.h"

// standard includes
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <limits.h>
#include <stdint.h>
#include <cstdio>

using namespace std;


//*************************************************
// Data type tydedef
//*************************************************
typedef uint32_t CHRPOS;
typedef uint16_t BINLEVEL;
typedef uint32_t BIN;
typedef uint16_t USHORT;
typedef uint32_t UINT;

//*************************************************
// Genome binning constants
//*************************************************

const BIN      _numBins   = 37450;
const BINLEVEL _binLevels = 7;

const BIN _binOffsetsExtended[] = {32678+4096+512+64+8+1, 4096+512+64+8+1, 512+64+8+1, 64+8+1, 8+1, 1, 0};

const USHORT _binFirstShift = 14;       /* How much to shift to get to finest bin. */
const USHORT _binNextShift  = 3;        /* How much to shift to get to next larger bin. */


//*************************************************
// Common data structures
//*************************************************

struct DEPTH {
    UINT starts;
    UINT ends;
};


struct BED {

    string chrom;
    CHRPOS start;
    CHRPOS end;
    string name;
    string score;
    string strand;

    double signalValue;
    double pValue;
    double qValue;
    int pointSource;

    // experimental fields for the FJOIN approach.
    bool   zeroLength;
    bool   added;
    bool   finished;
    // list of hits from another file.
    vector<BED> overlaps;

public:
    // constructors

    // Null
    BED()
    : chrom(""),
        start(0),
        end(0),
        name(""),
        score(""),
        strand(""),
        signalValue(0.0),
        pValue(0.0),
        qValue(0.0),
        pointSource(0),
        zeroLength(false),
        added(false),
        finished(false),
        overlaps()
    {}

    // broadPeak format
    BED(string chrom, CHRPOS start, CHRPOS end, string name,
            string score, string strand, double signalValue, double pValue, double qValue)
    : chrom(chrom),
        start(start),
        end(end),
        name(name),
        score(score),
        strand(strand),
        signalValue(signalValue),
        pValue(pValue),
        qValue(qValue),
        pointSource(),
        zeroLength(false),
        added(false),
        finished(false),
        overlaps()
    {}

    // narrowPeak format
    BED(string chrom, CHRPOS start, CHRPOS end, string name,
            string score, string strand, double signalValue, double pValue, double qValue, int pointSource)
    : chrom(chrom),
      start(start),
      end(end),
      name(name),
      score(score),
      strand(strand),
      signalValue(signalValue),
      pValue(pValue),
      qValue(qValue),
      pointSource(pointSource),
      zeroLength(false),
      added(false),
      finished(false),
      overlaps()
    {}

    int size() {
        return end-start;
    }

}; // BED


/*
    Structure for each end of a paired BED record
    mate points to the other end.
*/
struct MATE {
    BED bed;
    int lineNum;
    MATE *mate;
};


/*
    Structure for regular BED COVERAGE records
*/
struct BEDCOV {

    string chrom;

    // Regular BED fields
    CHRPOS start;
    CHRPOS end;
    string name;
    string score;
    string strand;

    double signalValue;
    double pValue;
    double qValue;
    int pointSource;

    // flag a zero-length feature
    bool   zeroLength;

    // Additional fields specific to computing coverage
    map<unsigned int, DEPTH> depthMap;
    unsigned int count;
    CHRPOS minOverlapStart;


    public:
    // constructors
    // Null
    BEDCOV()
    : chrom(""),
      start(0),
      end(0),
      name(""),
      score(""),
      strand(""),
      signalValue(0.0),
      pValue(0.0),
      qValue(0.0),
      pointSource(0),
      zeroLength(false),
      depthMap(),
      count(0),
      minOverlapStart(0)
    {}
};


/*
    Structure for BED COVERAGE records having lists of
    multiple coverages
*/
struct BEDCOVLIST {

    // Regular BED fields
    string chrom;
    CHRPOS start;
    CHRPOS end;
    string name;
    string score;
    string strand;

    double signalValue;
    double pValue;
    double qValue;
    int pointSource;

    // flag a zero-length feature
    bool   zeroLength;

    // Additional fields specific to computing coverage
    vector< map<unsigned int, DEPTH> > depthMapList;
    vector<unsigned int> counts;
    vector<CHRPOS> minOverlapStarts;


    public:
    // constructors
    // Null
    BEDCOVLIST()
    : chrom(""),
      start(0),
      end(0),
      name(""),
      score(""),
      strand(""),
      signalValue(0.0),
      pValue(0.0),
      qValue(0.0),
      pointSource(0),
      zeroLength(false),
      depthMapList(),
      counts(0),
      minOverlapStarts(0)
    {}
};


// enum to flag the state of a given line in a BED file.
enum BedLineStatus
{
    BED_INVALID = -1,
    BED_HEADER  = 0,
    BED_BLANK   = 1,
    BED_VALID   = 2
};

// enum to indicate the type of file we are dealing with
enum FileType
{
    BED_FILETYPE,
    GFF_FILETYPE,
    VCF_FILETYPE
};

//*************************************************
// Data structure typedefs
//*************************************************
typedef vector<BED>    bedVector;
typedef vector<BEDCOV> bedCovVector;
typedef vector<MATE> mateVector;
typedef vector<BEDCOVLIST> bedCovListVector;

typedef map<BIN, bedVector,    std::less<BIN> > binsToBeds;
typedef map<BIN, bedCovVector, std::less<BIN> > binsToBedCovs;
typedef map<BIN, mateVector, std::less<BIN> > binsToMates;
typedef map<BIN, bedCovListVector, std::less<BIN> > binsToBedCovLists;

typedef map<string, binsToBeds, std::less<string> >    masterBedMap;
typedef map<string, binsToBedCovs, std::less<string> > masterBedCovMap;
typedef map<string, binsToMates, std::less<string> > masterMateMap;
typedef map<string, binsToBedCovLists, std::less<string> > masterBedCovListMap;
typedef map<string, bedVector, std::less<string> >     masterBedMapNoBin;

// return the genome "bin" for a feature with this start and end
inline
BIN getBin(CHRPOS start, CHRPOS end) {
    --end;
    start >>= _binFirstShift;
    end   >>= _binFirstShift;

    for (register short i = 0; i < _binLevels; ++i) {
        if (start == end) return _binOffsetsExtended[i] + start;
        start >>= _binNextShift;
        end   >>= _binNextShift;
    }
    cerr << "start " << start << ", end " << end << " out of range in findBin (max is 512M)" << endl;
    return 0;
}

/****************************************************
// isInteger(s): Tests if string s is a valid integer
*****************************************************/
inline bool isInteger(const std::string& s) {
    int len = s.length();
    for (int i = 0; i < len; i++) {
        if (!std::isdigit(s[i])) return false;
    }
    return true;
}


// return the amount of overlap between two features.  Negative if none and the the
// number of negative bases is the distance between the two.
inline
int overlaps(CHRPOS aS, CHRPOS aE, CHRPOS bS, CHRPOS bE) {
    return min(aE, bE) - max(aS, bS);
}


// Ancillary functions
void splitBedIntoBlocks(const BED &bed, int lineNum, bedVector &bedBlocks);


// BED Sorting Methods
bool sortByChrom(const BED &a, const BED &b);
bool sortByStart(const BED &a, const BED &b);
bool sortBySizeAsc(const BED &a, const BED &b);
bool sortBySizeDesc(const BED &a, const BED &b);
bool sortByScoreAsc(const BED &a, const BED &b);
bool sortByScoreDesc(const BED &a, const BED &b);
bool byChromThenStart(BED const &a, BED const &b);



//************************************************
// BedFile Class methods and elements
//************************************************
class BedFile {

public:

    // Constructor
    BedFile(string &);

    // Destructor
    ~BedFile(void);

    // Open a BED file for reading (creates an istream pointer)
    void Open(void);

    // Close an opened BED file.
    void Close(void);

    // Get the next BED entry in an opened BED file.
    BedLineStatus GetNextBed (BED &bed, int &lineNum);

    int countLines();

    // load a BED file into a vector
    void loadBedFileIntoVector();

    // the bedfile with which this instance is associated
    string bedFile;
    unsigned int bedType;  // 3-6, 12 for BED
                            // 9 for GFF
    bool isZeroBased;

    // Main data structires used by BEDTools
    masterBedCovMap      bedCovMap;
    masterBedCovListMap  bedCovListMap;
    masterBedMap         bedMap;
    masterBedMapNoBin    bedMapNoBin;
    bedVector            bedList;

private:

    // data
    bool _isGff;
    bool _isVcf;
    bool _typeIsKnown;        // do we know the type?   (i.e., BED, GFF, VCF)
    FileType   _fileType;     // what is the file type? (BED? GFF? VCF?)
    istream   *_bedStream;
    string _bedLine;
    vector<string> _bedFields;

    void setZeroBased(bool zeroBased);
    void setGff (bool isGff);
    void setVcf (bool isVcf);
    void setFileType (FileType type);
    void setBedType (int colNums);

    /******************************************************
    Private definitions to circumvent linker issues with
    templated member functions.
    *******************************************************/

    /*
        parseLine: converts a lineVector into either BED or BEDCOV (templated, hence in header to avoid linker issues.)
    */
    template <typename T>
    inline BedLineStatus parseLine (T &bed, const vector<string> &lineVector, int &lineNum) {

        //char *p2End, *p3End, *p4End, *p5End;
        //long l2, l3, l4, l5;
        unsigned int numFields = lineVector.size();

        // bail out if we have a blank line
        if (numFields == 0) {
            return BED_BLANK;
        }

        if ((lineVector[0].find("track") == string::npos) && (lineVector[0].find("browser") == string::npos) && (lineVector[0].find("#") == string::npos) ) {

            if (numFields >= 3) {
                // line parsing for all lines after the first non-header line
                if (_typeIsKnown == true) {
                    switch(_fileType) {
                        case BED_FILETYPE:
                            if (parseBedLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
                        default:
                            printf("ERROR: file type encountered. Exiting\n");
                            exit(1);
                    }
                }
                // line parsing for first non-header line: figure out file contents
                else {
                    // it's BED format if columns 2 and 3 are integers
                    if (isInteger(lineVector[1]) && isInteger(lineVector[2])) {
                        setGff(false);
                        setZeroBased(true);
                        setFileType(BED_FILETYPE);
                        setBedType(numFields);       // we now expect numFields columns in each line
                        if (parseBedLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
                    }
                    else {
                        cerr << "Unexpected file format.  Please use tab-delimited BED, GFF, or VCF. " <<
                                "Perhaps you have non-integer starts or ends at line " << lineNum << "?" << endl;
                        exit(1);
                    }
                }
            }
            else {
                cerr << "It looks as though you have less than 3 columns at line: " << lineNum << ".  Are you sure your files are tab-delimited?" << endl;
                exit(1);
            }
        }
        else {
            lineNum--;
            return BED_HEADER;
        }
        // default
        return BED_INVALID;
    }


    /*
        parseBedLine: converts a lineVector into either BED or BEDCOV (templated, hence in header to avoid linker issues.)
    */
    template <typename T>
    inline bool parseBedLine (T &bed, const vector<string> &lineVector, int lineNum, unsigned int numFields) {

        // process as long as the number of fields in this
        // line matches what we expect for this file.
        if (numFields == this->bedType) {
            bed.chrom = lineVector[0];
            int i;
            i = atoi(lineVector[1].c_str());
            if (i<0) {
                 cerr << "Error: malformed BED entry at line " << lineNum << ". Start Coordinate detected that is < 0. Exiting." << endl;
                 exit(1);
            }
            bed.start = (CHRPOS)i;
            i = atoi(lineVector[2].c_str());
            if (i<0) {
                cerr << "Error: malformed BED entry at line " << lineNum << ". End Coordinate detected that is < 0. Exiting." << endl;
                exit(1);
            }
            bed.end = (CHRPOS)i;
            // handle starts == end (e.g., insertions in reference genome)
            if (bed.start == bed.end) {
                bed.start--;
                bed.end++;
                bed.zeroLength = true;
            }
            bed.name        =   lineVector[3];
            bed.score       =   lineVector[4];
            bed.strand      =   lineVector[5];
            bed.signalValue =   atof(lineVector[6].c_str());
            bed.pValue      =   atof(lineVector[7].c_str());
            bed.qValue      =   atof(lineVector[8].c_str());

            if (this->bedType == 10) {
                bed.pointSource = atoi(lineVector[9].c_str());
            }

            // sanity checks.
            if (bed.start <= bed.end) {
                return true;
            }
        }
        else if (numFields == 1) {
            cerr << "Only one BED field detected: " << lineNum << ".  Verify that your files are TAB-delimited.  Exiting..." << endl;
            exit(1);
        }
        else if ((numFields != this->bedType) && (numFields != 0)) {
            cerr << "Differing number of BED fields encountered at line: " << lineNum << ".  Exiting..." << endl;
            exit(1);
        }
        else if ((numFields < 3) && (numFields != 0)) {
            cerr << "TAB delimited BED file with at least 3 fields (chrom, start, end) is required at line: "<< lineNum << ".  Exiting..." << endl;
            exit(1);
        }
        return false;
    }
};

#endif /* BEDFILE_H */
