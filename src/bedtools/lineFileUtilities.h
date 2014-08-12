/*****************************************************************************
  parser.h

  (c) 2014 - Nikhil R Podduturi
  Cherry Lab, Stanford University

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/
#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <string.h>
#include <stdlib.h>
#include <sstream>

using namespace std;

// templated function to convert objects to strings
template <typename T>
inline
std::string ToString(const T & value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

// tokenize into a list of strings.
inline
void Tokenize(const string &str, vector<string> &elems, const string &delimiter = "\t")
{
    char* tok;
    char cchars [str.size()+1];
    char* cstr = &cchars[0];
    strcpy(cstr, str.c_str());
    tok = strtok(cstr, delimiter.c_str());
    while (tok != NULL) {
        elems.push_back(tok);
        tok = strtok(NULL, delimiter.c_str());
    }
}

// tokenize into a list of integers
inline
void Tokenize(const string &str, vector<int> &elems, const string &delimiter = "\t")
{
    char* tok;
    char cchars [str.size()+1];
    char* cstr = &cchars[0];
    strcpy(cstr, str.c_str());
    tok = strtok(cstr, delimiter.c_str());
    while (tok != NULL) {
        elems.push_back(atoi(tok));
        tok = strtok(NULL, delimiter.c_str());
    }
}

#endif /* PARSER_H */
