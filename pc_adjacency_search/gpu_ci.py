import numpy as np
from torch.multiprocessing import RawArray
from itertools import combinations, product

import globals

import time
import math
import gpucmiknn

def gpu_single(input_data):
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

    # Execute PC Algorithm
    lvls = range(globals.start_level,globals.max_sepset_size + 1)
    for lvl in lvls:
        print("Processing level: ",lvl)
        configs = set([(i, j, lvl) for i, j in product(cols_map, cols_map) if i != j and graph_array[i][j] == 1  and (graph_array[i]==1).sum() > lvl + 1])
        nr_edges = len(configs)
        if not configs:
            break
        result = []
        
        start = time.time()
        while len(configs) > 0 :
            config = configs.pop()
            res = gpu_single_test_edge(config)
            if res:
                result.append(res)
                try:
                    configs.remove((config[1],config[0],lvl))
                except:
                    pass


        end = time.time()
        print("Time for Level", lvl, ":", end-start, "Edges",nr_edges)

        for r in result:
            if r is not None:
                graph_array[r[0]][r[1]] = 0
                graph_array[r[1]][r[0]] = 0
                sepsets[(r[0], r[1])] = {'p_val': r[2], 'sepset': r[3]}
    return (graph_array, sepsets)

def gpu_single_test_edge(config):
    i = config[0]
    j = config[1]
    lvl = config[2]
    alpha = globals.alpha
    graph = np.frombuffer(globals.graph, dtype="int32").reshape((globals.vertices,
                                                                     globals.vertices))
    
    # unconditional
    if lvl < 1:
        p_val = gpu_single_pval_l0(i, j)
        if (p_val > alpha):
            return (i, j, p_val, [])
    # conditional
    else:
        candidates_1 = np.arange(globals.vertices)[(graph[i] == 1)]
        candidates_1 = np.delete(candidates_1, np.argwhere((candidates_1==i) | (candidates_1==j)))

        if (len(candidates_1) < lvl):
            return None
        for S in [list(c) for c in combinations(candidates_1, lvl)]:
            p_val = gpu_single_pval_ln(i, j, S)
            if (p_val > alpha):
                return (i, j, p_val, S)
          
    return None

def gpu_single_pval_l0(x_id, y_id):
    observations = np.frombuffer(globals.observations).reshape(globals.observations_shape)
    x = observations[:, [x_id]]
    y = observations[:, [y_id]]    
    
    k = int(globals.observations_shape[0] * 0.2) if globals.k_cmi == 'adaptive' else globals.k_cmi ### use 0.2 as factor from Runge paper

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N, 1))
    if y.ndim == 1:
        y = y.reshape((N, 1))
    data = np.concatenate((x, y), axis=1)
    data_t = np.transpose(data).astype(np.float32)
    pval = np.float32(gpucmiknn.pval_l0(data_t, k+1, globals.permutations))
    return pval

def gpu_single_pval_ln(x_id, y_id, z_ids):
    observations = np.frombuffer(globals.observations).reshape(globals.observations_shape)
    x = observations[:, [x_id]]
    y = observations[:, [y_id]]
    z = observations[:, z_ids]
    data = np.concatenate((x, y, z), axis=1).astype(np.float32)
    data_t = np.transpose(data).astype(np.float32)

    k = int(globals.observations_shape[0] * 0.2) if globals.k_cmi == 'adaptive' else globals.k_cmi ### use 0.2 as factor from Runge paper

    restricted_permutationN = gpucmiknn.rperm_multi(np.transpose(z).astype(np.float32), globals.permutations)
    x_permutationsN = np.ndarray(shape=(globals.permutations,len(x)), dtype='float32')
    for i in range(globals.permutations):
        x_permutationsN[i] = x[restricted_permutationN[i]].reshape(len(x))
    pval = np.float32(gpucmiknn.pval_ln(data_t, x_permutationsN, k+1, globals.k_perm+1, globals.permutations))

    return pval

# Rowwise on GPU
def gpu_row(input_data):
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
    # Execute PC Algorithm
    lvls = range(globals.start_level,globals.max_sepset_size + 1)
    for lvl in lvls:
        print("Processing level: ",lvl)
        graph_array_cp = np.copy(graph_array)
        configs = [(i, lvl) for i in cols_map if (graph_array[i]==1).sum() > lvl + 1]
        if not configs:
            break
        edges = [(i, j) for i, j in product(cols_map, cols_map) if i != j and graph_array[i][j] == 1  and (graph_array[i]==1).sum() > lvl + 1]
        result = []
        start = time.time()
        for config in configs:
            res = gpu_test_row(config, graph_array, graph_array_cp, cols_map)
            if res:
                for r in res:
                    graph_array_cp[r[0]][r[1]] = 0
                    graph_array_cp[r[1]][r[0]] = 0
                    sepsets[(r[0], r[1])] = {'p_val': r[2], 'sepset': r[3]}
        end = time.time()
        print("Time for Level", lvl, ":", end-start, "Edges",len(edges))

        np.copyto(graph_array,graph_array_cp)
    return (graph_array, sepsets)

def gpu_test_row(config, graph_array, graph_array_cp, cols_map):
    i = config[0]
    lvl = config[1]

    candidates_y = np.arange(globals.vertices)[(graph_array_cp[i] == 1)]
    candidates_y = np.delete(candidates_y, np.argwhere(candidates_y==i))
    if lvl == 0:
        return gpu_row_l0_pval(i, candidates_y)

    candidates_1 = np.arange(globals.vertices)[(graph_array[i] == 1)]
    candidates_1 = np.delete(candidates_1, np.argwhere(candidates_1==i))
    if len(candidates_1) <= lvl or len(graph_array_cp[i]) <= lvl:
        return None

    return gpu_row_pvals_ln(i, candidates_1, candidates_y, lvl, cols_map)

def nr_split_sep_sets(sepList, ydim):
    globals.permutations
    fix_mem = ((globals.observations_shape[0] * globals.observations_shape[1]) +
               (ydim * (globals.permutations + 1)) +
               ((ydim * 4))) * 4
    # substract a large buffer
    remaining_mem = globals.gpu_free_mem - fix_mem - 5000000000
    sep_mem = (len(sepList) * globals.observations_shape[0] * globals.permutations ) * 4 * 2
    nr_arrays = math.ceil(sep_mem / remaining_mem)
    return nr_arrays

def chunks(lst, n):
    """Return successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def gpu_row_pvals_ln(x_id, candidates, candidates_y, lvl, cols_map):
    ### get all data
    obs = np.frombuffer(globals.observations).reshape(globals.observations_shape)
    ### get all candidates
    sepList = np.array([list(c) for c in combinations(candidates, lvl)])
    ### split S in case of very large values
    splitted = np.array_split(sepList, nr_split_sep_sets(sepList, len(candidates))) if not globals.split_size else \
                chunks(sepList, globals.split_size)
    isSplitted = len(list(splitted)) > 1
    res = []
    for sList in splitted:
        if candidates_y.size == 0:
            break
        restricted_permutation_all = gpucmiknn.rperm_multi_all(obs.astype(np.float32), sList.astype(np.int), globals.permutations, x_id.item())
        k = int(globals.observations_shape[0] * 0.2) if globals.k_cmi == 'adaptive' else globals.k_cmi ### use 0.2 as factor from Runge paper
        sOfX, pvalOfX, candidates_y = gpucmiknn.pval_ln_row(obs.astype(np.float32), x_id.item(), restricted_permutation_all, k+1, globals.k_perm+1, globals.permutations, lvl, sList.astype(np.int), globals.alpha, np.array(candidates_y), len(candidates), isSplitted)
        for j in range(0,len(cols_map)):
            if pvalOfX[j] > globals.alpha:
                res.append((x_id, j, pvalOfX[j], sOfX[j]))
    return res

def gpu_row_l0_pval(x_id, candidates_y):
    ### get all data
    obs = np.frombuffer(globals.observations).reshape(globals.observations_shape)

    k = int(globals.observations_shape[0] * 0.2) if globals.k_cmi == 'adaptive' else globals.k_cmi ### use 0.2 as factor from Runge paper

    pvalOfX = gpucmiknn.pval_l0_row(obs.astype(np.float32), x_id.item(), k+1, globals.permutations, np.array(candidates_y))
    return [(x_id, candidates_y[j], pvalOfX[j],[]) for j in range(0,len(candidates_y)) if pvalOfX[j] > globals.alpha]

