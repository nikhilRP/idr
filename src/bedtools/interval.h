/*****************************************************************************
interval.h
(c) 2012 - Ryan M. Layer
Hall Laboratory
Quinlan Laboratory
Department of Computer Science
Department of Biochemistry and Molecular Genetics
Department of Public Health Sciences and Center for Public Health Genomics,
University of Virginia
rl6sf@virginia.edu

Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#ifndef __INTERVAL_H__
#define __INTERVAL_H__

struct interval {
	// order is hack to maintin the position of the interval in the orginal
	// data file
	unsigned int start, end, order;
};

#endif
