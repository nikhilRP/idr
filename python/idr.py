import os, sys

import ctypes

class c_OptimizationRV(ctypes.Structure):
    _fields_ = [("n_iters", ctypes.c_int), 
                ("rho", ctypes.c_double), 
                ("p", ctypes.c_double)]
                
C_em_gaussian = ctypes.cdll.LoadLibrary(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                 "IDR_parameter_estimation.so")).em_gaussian
C_em_gaussian.restype = c_OptimizationRV

import numpy

def mean(items):
    items = list(items)
    return sum(items)/float(len(items))

from collections import namedtuple, defaultdict, OrderedDict
from itertools import chain
Peak = namedtuple('Peak', ['chrm', 'strand', 'start', 'stop', 'signal'])

VERBOSE = False
QUIET = False

IGNORE_NONOVERLAPPING_PEAKS = False

def em_gaussian(ranks_1, ranks_2):
    n = len(ranks_1)
    assert( n == len(ranks_1) == len(ranks_2) )
    localIDR = numpy.zeros(n, dtype='d')
    rv = C_em_gaussian(
        ctypes.c_int(n), 
        ranks_1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ranks_2.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        localIDR.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        (ctypes.c_int(1) if VERBOSE else ctypes.c_int(0)) )
    n_iters, rho, p = rv.n_iters, rv.rho, rv.p
    return (n_iters, rho, p), localIDR

def load_bed(fp, signal_type):
    signal_index = {"signal.value": 6, "p.value": 7, "q.value": 8}[signal_type]
    grpd_peaks = defaultdict(list)
    for line in fp:
        if line.startswith("#"): continue
        if line.startswith("track"): continue
        data = line.split()
        signal = float(data[signal_index])
        if signal < 0: 
            raise ValueError("Invalid {}: {:e}".format(signal_type, signal))
        peak = Peak(data[0], data[5], int(data[1]), int(data[2]), signal )
        grpd_peaks[(peak.chrm, peak.strand)].append(peak)
    return grpd_peaks

def merge_peaks_in_contig(s1_peaks, s2_peaks, pk_agg_fn):
    """Merge peaks in a single contig/strand.
    
    returns: The merged peaks. 
    """
    # merge and sort all peaks, keeping track of which sample they originated in
    all_intervals = sorted(chain(
            ((pk.start,pk.stop,pk.signal,1) for i, pk in enumerate(s1_peaks)),
            ((pk.start,pk.stop,pk.signal,2) for i, pk in enumerate(s2_peaks))))
    
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
        if IGNORE_NONOVERLAPPING_PEAKS:
            if any(0 == len(peaks) for peaks in grpd_peaks.values()):
                continue

        s1, s2 = (pk_agg_fn(pk[2] for pk in pks) for pks in grpd_peaks.values())
        merged_pk = (pk_start, pk_stop, s1, s2, grpd_peaks)
        merged_pks.append(merged_pk)
    
    return merged_pks

def merge_peaks(s1_peaks, s2_peaks, pk_agg_fn):
    """Merge peaks over all contig/strands
    
    """
    contigs = sorted(set(chain(s1_peaks.keys(), s2_peaks.keys())))
    merged_peaks = []
    for key in contigs:
        # since s*_peaks are default dicts, it will never raise a key error, 
        # but instead return an empty list which is what we want
        merged_peaks.extend(
            key + pk for pk in merge_peaks_in_contig(
                s1_peaks[key], s2_peaks[key], pk_agg_fn))
    
    merged_peaks.sort(key=lambda x:pk_agg_fn((x[4],x[5])), reverse=True)
    return merged_peaks

def build_rank_vectors(merged_peaks):
    # allocate memory for the ranks vector
    s1 = numpy.zeros(len(merged_peaks))
    s2 = numpy.zeros(len(merged_peaks))
    # add the signal
    for i, x in enumerate(merged_peaks):
        s1[i], s2[i] = x[4], x[5]
    
    rank1 = numpy.lexsort((numpy.random.random(len(s1)), -s1)).argsort()
    rank2 = numpy.lexsort((numpy.random.random(len(s2)), -s2)).argsort()
    
    return numpy.array(rank1, dtype='i'), numpy.array(rank2, dtype='i')

def build_idr_output_line(
    contig, strand, signals, merged_peak, IDR, localIDR):
    rv = [contig,]
    for signal, key in zip(signals, (1,2)):
        if len(merged_peak[key]) == 0: 
            rv.extend(("-1", "-1"))
        else:
            rv.append( "%i" % min(x[0] for x in merged_peak[key]))
            rv.append( "%i" % max(x[1] for x in merged_peak[key]))
        rv.append( "%.5f" % signal )
    
    rv.append("%.5f" % IDR)
    rv.append("%.5f" % localIDR)
    rv.append(strand)
        
    return "\t".join(rv)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="""
Program: IDR (Irreproducible Discovery Rate)
Version: {PACKAGE_VERSION}\n
Contact: Nikhil R Podduturi <nikhilrp@stanford.edu>
         Nathan Boley <npboley@gmail.com>

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
    
    parser.add_argument( '--use-nonoverlapping-peaks', 
                         action="store_true", default=False,
        help='Use peaks without an overlapping match, setting the value to 0 for signal and 1 for p/qvalue.')
    
    parser.add_argument( '--peak-merge-method', 
                         choices=["sum", "avg", "min", "max"], default=None,
        help="Which method to use for merging peaks. Default: 'mean' for signal, 'min' for p/q-value.")
    
    parser.add_argument( '--verbose', action="store_true", default=False, 
                         help="Print out additional debug information")
    parser.add_argument( '--quiet', action="store_true", default=False, 
                         help="Don't print any status messages")

    args = parser.parse_args()

    global VERBOSE
    if args.verbose: VERBOSE = True 

    global QUIET
    if args.quiet: 
        QUIET = True 
        VERBOSE = False

    global IGNORE_NONOVERLAPPING_PEAKS
    IGNORE_NONOVERLAPPING_PEAKS = not args.use_nonoverlapping_peaks

    # decide what aggregation function to use for peaks that need to be merged
    if args.peak_merge_method == None:
        peak_merge_fn = {"signal.value": mean, "q.value": mean, "p.value": mean}[
            args.rank]
    else:
        peak_merge_fn = {"sum": sum, "avg": mean, "min": min, "max": max}[
            args.peak_merge_method]

    return args, peak_merge_fn

def log(msg, level=None):
    if QUIET: return
    if level == None or (level == 'VERBOSE' and VERBOSE):
        print(msg, file=sys.stderr)

def main():
    args, peak_merge_fn = parse_args()
    
    # load the peak files
    log("Loading the peak files", 'VERBOSE');
    f1 = load_bed(args.a, args.rank)
    f2 = load_bed(args.b, args.rank)

    # build a unified peak set
    log("Merging peaks", 'VERBOSE');
    merged_peaks = merge_peaks(f1, f2, peak_merge_fn)
    
    # build the ranks vector
    log("Ranking peaks", 'VERBOSE');
    s1, s2 = build_rank_vectors(merged_peaks)
    
    if( len(merged_peaks) < 20 ):
        error_msg = "Peak files must contain at least 20 peaks post-merge"
        error_msg += "\nHint: Merged peaks were written to the output file"
        for pk in merged_peaks: print( pk, file=args.output_file )
        raise ValueError(error_msg)
    
    # fit the model parameters    
    # (e.g. call the local idr C estimation code)
    log("Fitting the model parameters", 'VERBOSE');
    (n_iter, rho, p), IDRs = em_gaussian(s1, s2)
    
    log("Finished running IDR on the datasets");
    log("Final P value = %.15f" % p);
    log("Final rho value = %.15f" % rho);
    log("Total iterations of EM - %i" % n_iter);
    
    # build the global IDR array
    log("Building the global IDR array", 'VERBOSE');
    merged_data = sorted(zip(IDRs, merged_peaks))
    localIDRs = [merged_data[0][0],]
    for i, (IDR, merged_peak) in enumerate(merged_data[1:]):
        localIDRs.append( (IDR + localIDRs[i])/(i+2) )
    
    # write out the ouput
    log("Writing results to file", "VERBOSE");
    num_peaks_passing_thresh = 0
    for localIDR, (IDR, merged_peak) in zip(
            localIDRs, merged_data):
        # skip peaks with global idr values below the threshold
        if IDR > args.idr: continue
        num_peaks_passing_thresh += 1
        opline = build_idr_output_line(
            merged_peak[0], merged_peak[1], 
            merged_peak[4:6], 
            merged_peak[6], IDR, localIDR )
        print( opline, file=args.output_file )

    log("Number of peaks passing IDR cutoff of {} - {}\n".format(
            args.idr, num_peaks_passing_thresh))
    args.output_file.close()

if __name__ == '__main__':
    main()
