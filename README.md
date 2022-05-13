# gpucmiknn

A research implementation of a concept for GPU-accelerated information theoretic causal discovery, based upon the CMIknn test for conditional independence [^Runge2018].
The gpucmiknn implementation can be extended to handle different CMI estimators that build upon the concept of knn searches. 


## Installation
1. setup conda environment
    * install conda
    * ho to: https://rapids.ai/start.html#rapids-release-selector and configure the desired Rapids environment build command
    * execute the corresponding build command
    * activate the conda environment
2. build cuda code:
    * in parent directory of this repo: <br>
    ```cd pc_adjacency_search```
    * adjust Makefile.config, e.g. (remaining lines can be commented out): <br>
    ```
    ANACONDA_HOME := $(HOME)/anaconda3/envs/rapids-21.08
    PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
    $(ANACONDA_HOME)/include/python3.7m \
    $(ANACONDA_HOME)/lib/python3.7/site-packages/numpy/core/include/
    PYTHON_LIB := $(ANACONDA_HOME)/lib
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
    CUDA_DIR := /usr/local/cuda
    BUILD_DIR := build
    ```
    * build the code: <br>
    ```make```  
    * tested with python 3.8
3. install python dependencies:
	```pip install -r requirements.txt```

## Execution
   In parent directory of this repo: <br>
    ```
    cd pc_adjacency_search
	python main.py -i ./data/coolingData.csv --permutations 100 --process_count 1 -a 0.01 --par_strategy 2 -k 7
	```

## Parameters

The following options are available:
| Parameter  | Default |  Description |
|---|---|---|
| -i |  | Path to input file in .csv format. |
| -a | 0.05  | Sets the significance level used within PC algorithm. |
| -l | None  | Gives the max level for the PC algorithm (level of the pc algorithm is <= max level) |
| -k | adaptive  | k-nearest neighbors during CMI estimation. Adaptive, sets the parameter to 0.2 the sample size. |
| --permutations | 50   |  Number of Permutations used for the CI Test.|
| --par_strategy |  | Flag indicate the parallel hardware used: 1 - CPU-based execution; 2 - GPUKNNCMI-Single; 3 GPUKNNCMI-Parallel |
| --process_count | 2   |  Number of parallel processes used during adjacency search for CPU-based execution |
| -b | None  | Blocks during block-wise processing of GPUKNNCMI-Parallel. Default setting calculates the blocks on encountering memory pressure due to large numbers of separation set candidates. |


## Contributor
* [Constantin Lange](https://github.com/constantin-lange)
* [Christopher Hagedorn](https://github.com/ChristopherSchmidt89)

## License

GPL-3

## References
[^Runge2018]: CMIknn: J. Runge (2018): Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics. http://proceedings.mlr.press/v84/runge18a.html

