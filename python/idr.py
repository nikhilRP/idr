import os, sys

import ctypes
C_em_gaussian = ctypes.cdll.LoadLibrary("./IDR_parameter_estimation.so").em_gaussian

import numpy

from collections import namedtuple, defaultdict
from itertools import chain
Peak = namedtuple('Peak', ['chrm', 'strand', 'start', 'stop', 'signal'])

def em_gaussian(ranks_1, ranks_2):
    n = len(ranks_1)
    assert( n == len(ranks_1) == len(ranks_2) )
    localIDR = numpy.zeros(n, dtype='d')
    C_em_gaussian(
        ctypes.c_int(n), 
        ranks_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ranks_2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        localIDR.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  )
    return localIDR

def load_bed(fname):
    grpd_peaks = defaultdict(list)
    with open(fname) as fp:
        for line in fp:
            if line.startswith("#"): continue
            if line.startswith("track"): continue
            data = line.split()
            peak = Peak(data[0], data[5], int(data[1]), int(data[2]), float(data[6]))
            grpd_peaks[(peak.chrm, peak.strand)].append(peak)
    return grpd_peaks

def merge_peaks_in_contig(s1_peaks, s2_peaks):
    """Merge peaks in a single contig/strand.
    
    """
    # merge and sort all peaks, keeping track of which sample they originated inx
    all_intervals = sorted(chain(
            ((pk.start, pk.stop, pk.signal, 1) for pk in s1_peaks),
            ((pk.start, pk.stop, pk.signal, 2) for pk in s2_peaks)))

    # grp overlapping intervals. Since they're already sorted, all we need
    # to do is check if the current interval overlaps the previous interval
    grpd_intervals = [[],]
    curr_start, curr_stop = all_intervals[0][:2]
    for x in all_intervals:
        if x[0] < curr_stop:
            curr_stop = max(x[1], curr_stop)
            grpd_intervals[-1].append(x)
        else:
            curr_start, curr_stop = x[:2]
            grpd_intervals.append([x,])

    # build the unified peak list, setting the score to 
    # zero if it doesn't exist in both replicates
    pks = []
    for intervals in grpd_intervals:
        pk_start = min(x[0] for x in intervals)
        pk_stop = max(x[1] for x in intervals)
        s1 = sum( x[2] for x in intervals if x[3] == 1 )
        s2 = sum( x[2] for x in intervals if x[3] == 2 )
        if s1 == 0 or s2 == 0: continue
        pks.append((pk_start, pk_stop, s1, s2))
    
    return pks

def merge_peaks(s1_peaks, s2_peaks):
    """Merge peaks over all contig/strands
    
    """
    contigs = sorted(set(chain(s1_peaks.keys(), s2_peaks.keys())))
    merged_peaks = []
    for key in contigs:
        # since s*_peaks are default dicts, it will never raise a key error, 
        # but instead return an empty list which is what we want
        merged_peaks.extend(
            key + pk for pk in merge_peaks_in_contig(s1_peaks[key], s2_peaks[key]))
    return merged_peaks

def build_rank_vectors(merged_peaks):
    # allocate memory for the ranks vector
    s1 = numpy.zeros(len(merged_peaks))
    s2 = numpy.zeros(len(merged_peaks))
    # add the signal
    for i, x in enumerate(merged_peaks):
        s1[i], s2[i] = x[4], x[5]
    # build hte ranks - we add uniform random noise to break ties
    s1 = numpy.array((s1.argsort() + numpy.random.random(len(merged_peaks))).argsort(), dtype='d')
    s2 = numpy.array((s2.argsort() + numpy.random.random(len(merged_peaks))).argsort(), dtype='d')
    return s1, s2

def main():
    # load the peak files
    f1 = load_bed(sys.argv[1])
    f2 = load_bed(sys.argv[2])

    # build a unified peak set
    merged_peaks = merge_peaks(f1, f2)
    
    # build the ranks vector
    s1, s2 = build_rank_vectors(merged_peaks)

    # fit the model parameters    
    # (e.g. call the local idr C estimation code)
    localIDR = em_gaussian(s1, s2)
    pass
    
    # write out the output 
    pass

if __name__ == '__main__':
    main()
