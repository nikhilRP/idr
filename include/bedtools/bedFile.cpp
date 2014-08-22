#include "bedFile.h"

/***********************************************
Sorting comparison functions
************************************************/
bool sortByChrom(BED const &a, BED const &b) {
    if (a.chrom < b.chrom) return true;
    else return false;
};

bool sortByStart(const BED &a, const BED &b) {
    if (a.start < b.start) return true;
    else return false;
};

bool sortBySizeAsc(const BED &a, const BED &b) {

    CHRPOS aLen = a.end - a.start;
    CHRPOS bLen = b.end - b.start;

    if (aLen < bLen) return true;
    else return false;
};

bool sortBySizeDesc(const BED &a, const BED &b) {

    CHRPOS aLen = a.end - a.start;
    CHRPOS bLen = b.end - b.start;

    if (aLen > bLen) return true;
    else return false;
};

bool sortByScoreAsc(const BED &a, const BED &b) {
    if (a.score < b.score) return true;
    else return false;
};

bool sortByScoreDesc(const BED &a, const BED &b) {
    if (a.score > b.score) return true;
    else return false;
};

bool byChromThenStart(BED const &a, BED const &b) {

    if (a.chrom < b.chrom) return true;
    else if (a.chrom > b.chrom) return false;

    if (a.start < b.start) return true;
    else if (a.start >= b.start) return false;

    return false;
};


/*******************************************
Class methods
*******************************************/

// Constructor
BedFile::BedFile(string &bedFile)
: bedFile(bedFile),
  _typeIsKnown(false)
{}

// Destructor
BedFile::~BedFile(void) {
}


void BedFile::Open(void) {

    _bedFields.reserve(12);

    if (bedFile == "stdin" || bedFile == "-") {
        _bedStream = &cin;
    }
    else {
        _bedStream = new ifstream(bedFile.c_str(), ios::in);

        if ( !(_bedStream->good()) ) {
            cerr << "Error: The requested bed file (" << bedFile << ") could not be opened. Exiting!" << endl;
            exit (1);
        }
    }
}


// Close the BED file
void BedFile::Close(void) {
    if (bedFile != "stdin" && bedFile != "-") delete _bedStream;
}


BedLineStatus BedFile::GetNextBed(BED &bed, int &lineNum) {

    // make sure there are still lines to process.
    // if so, tokenize, validate and return the BED entry.
    _bedFields.clear();
    if (_bedStream->good()) {
        // parse the bedStream pointer
        getline(*_bedStream, _bedLine);
        lineNum++;

        // split into a string vector.
        Tokenize(_bedLine, _bedFields);

        // load the BED struct as long as it's a valid BED entry.
        return parseLine(bed, _bedFields, lineNum);
    }

    // default if file is closed or EOF
    return BED_INVALID;
}

void BedFile::setZeroBased(bool zeroBased) { this->isZeroBased = zeroBased; }

void BedFile::setGff (bool gff) { this->_isGff = gff; }


void BedFile::setVcf (bool vcf) { this->_isVcf = vcf; }


void BedFile::setFileType (FileType type) {
    _fileType    = type;
    _typeIsKnown = true;
}


void BedFile::setBedType (int colNums) {
    bedType = colNums;
}


int BedFile::countLines() {

    Open();
    int n = 0;
    while( getline(*_bedStream, _bedLine) ) {
        n++;
    }
    Close();
    return n;
}

void BedFile::loadBedFileIntoVector() {

    BED bedEntry, nullBed;
    int lineNum = 0;
    BedLineStatus bedStatus;

    bedList.reserve(100000000);

    Open();

    while ((bedStatus = this->GetNextBed(bedEntry, lineNum)) != BED_INVALID) {

        if (bedStatus == BED_VALID) {
            bedList.push_back(bedEntry);
            bedEntry = nullBed;
        }
    }
    Close();
}
