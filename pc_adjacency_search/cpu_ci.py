import numpy as np
from itertools import combinations, product
from torch.multiprocessing import Array, RawArray, Pool

import globals
import time

def cpu_adj(input_data): 
    # Prepare Data
    row_count = input_data.shape[0]
    col_count = input_data.shape[1]
    cols_map = np.arange(col_count)
    observations_raw = RawArray('d', row_count * col_count)
    observations_array = np.frombuffer(observations_raw, dtype=np.float64).reshape(input_data.shape)
    np.copyto(observations_array, input_data)
    globals.observations = observations_raw
    globals.observations_shape = input_data.shape

    graph_raw = RawArray('i', np.ones(col_count*col_count).astype(int))
    graph_array = np.frombuffer(graph_raw, dtype="int32").reshape((col_count, col_count))
    sepsets = {}
    globals.graph = graph_raw
    graph_raw_cp = Array('i', np.ones(col_count*col_count).astype(int),lock=True)
    graph_array_cp = np.frombuffer(graph_raw_cp.get_obj(), dtype='i').reshape((col_count, col_count))
    globals.graph_cp = graph_array_cp


    # Execute PC Algorithm
    lvls = range(globals.start_level,globals.max_sepset_size + 1)
    for lvl in lvls:
        print("Processing level: ",lvl)
        configs = [(i, j, lvl) for i, j in product(cols_map, cols_map) if i != j and graph_array[i][j] == 1 and (graph_array[i]==1).sum() > lvl + 1]
        if not configs:
            break
        ### LIMIT Configs to 20 for CItest measurements
        #configs = configs[:20]
        result = []
        start = 0
        end = 0
        if globals.process_count > 1:
            print("Processing in parallel")
            with Pool(processes=globals.process_count) as pool:
                print("starting time measurement")
                start = time.time()
                result = pool.map(cpu_test_edge, configs)
        else:
            print("Processing single-threaded")
            start = time.time()
            for config in configs:
                result.append(cpu_test_edge(config))
        end = time.time()
        print("Time for Level", lvl, ":", end-start, "Edges",len(configs))
        for r in result:
            if r is not None:
                sepsets[(r[0], r[1])] = {'p_val': r[2], 'sepset': r[3]}
        np.copyto(graph_array, graph_array_cp)
    return (graph_array, sepsets)

def cpu_test_edge(config):
    i = config[0]
    j = config[1]
    lvl = config[2]
    if globals.graph_cp[i][j] == 0:
        ### early termination as pair has been found independent by previous worker
        return None
    alpha = globals.alpha
    graph = np.frombuffer(globals.graph, dtype="int32").reshape((globals.vertices,
                                                                     globals.vertices))
    # unconditional
    if lvl < 1:
        ### Use cmiknn from tigramite here
        observations = np.frombuffer(globals.observations).reshape(globals.observations_shape)
        (val,p_val) = globals.ci_test.run_test_raw(observations[:, [i]], observations[:, [j]])
        if (p_val > alpha):
            # mark pair of variables as independent for (i,j) and (j,i)
            globals.graph_cp[i][j] = 0
            globals.graph_cp[j][i] = 0
            return (i, j, p_val, [])
    # conditional
    else:
        candidates_1 = np.arange(globals.vertices)[(graph[i] == 1)]
        candidates_1 = np.delete(candidates_1, np.argwhere((candidates_1==i) | (candidates_1==j)))
        if (len(candidates_1) < lvl):
            return None
        for S in [list(c) for c in combinations(candidates_1, lvl)]:
            ### Use cmiknn from tigramite here
            observations = np.frombuffer(globals.observations).reshape(globals.observations_shape)
            (val,p_val) = globals.ci_test.run_test_raw(observations[:, [i]], observations[:, [j]], observations[:, S])
            if (p_val > alpha):
                # mark pair of variables as independent for (i,j) and (j,i)
                globals.graph_cp[i][j] = 0
                globals.graph_cp[j][i] = 0
                return (i, j, p_val, S)
          
    return None
