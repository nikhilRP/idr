import os, sys

import ctypes

class c_OptimizationRV(ctypes.Structure):
    _fields_ = [("n_iters", ctypes.c_int), ("rho", ctypes.c_double), ("p", ctypes.c_double)]
                
C_em_gaussian = ctypes.cdll.LoadLibrary("./IDR_parameter_estimation.so").em_gaussian
C_em_gaussian.restype = c_OptimizationRV

import numpy

def mean(items):
    return sum(items)/float(len(items))

from collections import namedtuple, defaultdict, OrderedDict
from itertools import chain
Peak = namedtuple('Peak', ['chrm', 'strand', 'start', 'stop', 'signal'])

VERBOSE = False

def em_gaussian(ranks_1, ranks_2):
    n = len(ranks_1)
    assert( n == len(ranks_1) == len(ranks_2) )
    localIDR = numpy.zeros(n, dtype='d')
    rv = C_em_gaussian(
        ctypes.c_size_t(n), 
        ranks_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ranks_2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        localIDR.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  )
    n_iters, rho, p = rv.n_iters, rv.rho, rv.p
    return (n_iters, rho, p), localIDR

def load_bed(fp):
    grpd_peaks = defaultdict(list)
    for line in fp:
        if line.startswith("#"): continue
        if line.startswith("track"): continue
        data = line.split()
        peak = Peak(data[0], data[5], int(data[1]), int(data[2]), float(data[6]))
        grpd_peaks[(peak.chrm, peak.strand)].append(peak)
    return grpd_peaks

def merge_peaks_in_contig(s1_peaks, s2_peaks):
    """Merge peaks in a single contig/strand.
    
    returns: The merged peaks. 
    """
    # merge and sort all peaks, keeping track of which sample they originated inx
    all_intervals = sorted(chain(
            ((pk.start, pk.stop, pk.signal, 1) for i, pk in enumerate(s1_peaks)),
            ((pk.start, pk.stop, pk.signal, 2) for i, pk in enumerate(s2_peaks))))
    
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
    merged_pks = []
    for intervals in grpd_intervals:
        # grp peaks by their source, and calculate the merged
        # peak boundaries
        grpd_peaks = OrderedDict(((1, []), (2, [])))
        pk_start, pk_stop = 1e9, -1
        for x in intervals:
            pk_start = min(x[0], pk_start)
            pk_stop = max(x[0], pk_stop)
            grpd_peaks[x[3]].append(x)

        # skip regions that dont have a peak in all replicates
        if any(0 == len(peaks) for peaks in grpd_peaks.values()):
            continue

        s1, s2 = (sum(pk[2] for pk in pks) for pks in grpd_peaks.values())
        merged_pk = (pk_start, pk_stop, s1, s2, grpd_peaks)
        merged_pks.append(merged_pk)
    
    return merged_pks

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

def build_idr_output_line(contig, strand, signals, merged_peak, localIDR, globalIDR):
    rv = [contig,]
    for signal, key in zip(signals, (1,2)):
        rv.append( "%i" % min(x[0] for x in merged_peak[key]))
        rv.append( "%i" % max(x[1] for x in merged_peak[key]))
        rv.append( "%.5f" % signal )
    
    rv.append("%.5f" % globalIDR)
    rv.append("%.5f" % localIDR)
    rv.append(strand)
    
    return "\t".join(rv)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="""
Program: IDR (Irreproducible Discovery Rate)
Version: {PACKAGE_VERSION}\n
Contact: Nikhil R Podduturi <nikhilrp@stanford.edu>, Nathan Boley <npboley@gmail.com>

""")

    parser.add_argument( '-a', type=argparse.FileType("r"), required=True,
        help='narrowPeak or broadPeak file containing peaks from sample 1.')

    parser.add_argument( '-b', type=argparse.FileType("r"), required=True,
        help='narrowPeak or broadPeak file containing peaks from sample 2.')

    default_ofname = "idrValues.txt"
    parser.add_argument( '--output-file', "-o", type=argparse.FileType("w"), 
                         default=open(default_ofname, "w"), 
        help='File to write output to. default: {}'.format(default_ofname))

    parser.add_argument( '--idr', "-i", type=float, default=1.0, 
        help='Only report peaks with a global idr threshold below this value. Default: report all peaks')

    parser.add_argument( '--rank', default="signal.value",
                         choices=["signal.value", "p.value", "q.value"],
                         help='Type of ranking measure to use.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # load the peak files
    f1 = load_bed(args.a)
    f2 = load_bed(args.b)

    # build a unified peak set
    merged_peaks = merge_peaks(f1, f2)
    
    # build the ranks vector
    s1, s2 = build_rank_vectors(merged_peaks)

    # fit the model parameters    
    # (e.g. call the local idr C estimation code)
    (n_iter, rho, p), localIDRs = em_gaussian(s1, s2)
    
    print("Finished running IDR on the datasets");
    print("Final P value = %.15f" % p);
    print("Final rho value = %.15f" % rho);
    print("Total iterations of EM - %i" % n_iter);
    
    # build the global IDR array
    merged_data = sorted(zip(localIDRs, merged_peaks))
    globalIDRs = [merged_data[0][0],]
    for i, (localIDR, merged_peak) in enumerate(merged_data[1:]):
        globalIDRs.append( (localIDR + globalIDRs[i])/(i+1) )

    # write out the ouput
    for globalIDR, (localIDR, merged_peak) in zip(globalIDRs, merged_data):
        # skip peaks with global idr values below the threshold
        if globalIDR > args.idr: continue
        opline = build_idr_output_line(merged_peak[0], merged_peak[1], merged_peak[4:6], merged_peak[6], localIDR, globalIDR)
        print( opline, file=args.output_file )
    
    args.output_file.close()

if __name__ == '__main__':
    main()
