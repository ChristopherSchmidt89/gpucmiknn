import argparse
import numpy as np
import os
from gpu_ci import gpu_single, gpu_row
from cpu_ci import cpu_adj
from tigramite.independence_tests import CMIknn
import globals

import gpucmiknn

import sys

# packages for shd
import cdt
import networkx

parser = argparse.ArgumentParser("Parallel PC Algorithm on Mixed Data")
parser.add_argument('-a', '--alpha', help='Signficance Level used for CI Test', default=0.05, type=float)
parser.add_argument('-l', '--level', help='Maximum Level of the Run', default=None, type=int)
parser.add_argument('--process_count', help='Number of Processes used in the Run - Main process excluded', default=2, type=int)
parser.add_argument('--permutations', help='Number of Permutations used for a CI Test', default=50, type=int)
parser.add_argument('--par_strategy', help='Specify Parallelization Strategy: 1 -> CPU-Based | 2 -> Single GPU | 3 Rowwise GPU', type=int)
parser.add_argument('-s', '--start_level', help='Execute starting with level s for benchmarking tests with certain sepset size ', default=0, type=int)
parser.add_argument('-k', '--kcmi', help='KNN during cmi estimation. Default is adaptive, which is fixed to 7 for mesner, samples*0.2 for Runge or sqrt(samples)/5 for GKOV', type=int)
parser.add_argument('-b', '--block_size', help='Number of separation sets that are blocked together during rowwise processing of lvl > 0. Default is None and calculates the factor on encountering memory pressure due to large numbers of separation set candidates', default=None, type=int)
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input_file', help='Input File Name', required=True)

if __name__ == '__main__':
    # Call Parallel PC Algorithm Skeleton Estimation
    args = parser.parse_args()

    input_data = np.genfromtxt(args.input_file, delimiter=",", skip_header=1)
    np.set_printoptions(threshold=sys.maxsize)
    globals.init()
    globals.alpha = args.alpha
    globals.vertices = input_data.shape[1]
    globals.permutations = args.permutations
    globals.process_count = args.process_count
    globals.max_sepset_size = input_data.shape[1] - 2 if args.level is None else min(args.level, input_data.shape[1] - 2)

    globals.start_level = args.start_level

    globals.split_size = args.block_size
    if args.kcmi:
        globals.k_cmi = args.kcmi
    else:
        globals.k_cmi = 'adaptive'

    # Execption for Runge test for paper
    if (args.par_strategy == 1):
        globals.ci_test = CMIknn(knn=int(input_data.shape[0] * 0.2) if globals.k_cmi == 'adaptive' else globals.k_cmi,
                 shuffle_neighbors=globals.k_perm,
                 significance='shuffle_test',
                 sig_samples = globals.permutations,
                 workers=-1)
    else:
        globals.ci_test = None

    print("Starting level:", globals.start_level)

    skeleton = None
    sepsets = None
    if args.par_strategy == 2 or args.par_strategy == 3:
        globals.gpu_free_mem = gpucmiknn.init_gpu()
        print("init cuda")

    if (args.par_strategy == 3):
        print("Execution Rowwise on GPU")
        skeleton, sepsets = gpu_row(input_data)
    if (args.par_strategy == 2):
        print("Execution Single on GPU")
        skeleton, sepsets = gpu_single(input_data)
    if (args.par_strategy == 1):
        print("CPU only")
        skeleton, sepsets = cpu_adj(input_data)

    print(skeleton, sepsets)
    
