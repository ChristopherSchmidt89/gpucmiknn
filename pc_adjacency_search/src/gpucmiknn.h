float pval_l0_cuda_shared(const float*  obs,
                          size_t        obs_count,
                          size_t        dim,
                          int           k,
                          size_t        permutations);

float pval_ln_cuda(const float*  obs,
                   const float*  x_permutations,
                   size_t        obs_count,
                   size_t        dim,
                   int           k,
                   int           k_perm,
                   size_t        permutations);

void pval_l0_row_cuda(const float* data_c,
                      size_t obs_count,
                      int k,
                      size_t permutations,
                      int x_id,
                      size_t varCount,
                      float* pvalOfX,
                      int* candidates_c,
                      size_t yDim);

void pval_ln_row_cuda(const float* data_c, const float* x_permutations_c, size_t data_height,
                    int k, int k_perm, size_t permutations, int x_id, size_t varCount, int lvl, int* sList_c, size_t sEntries,
                    int* sOfX, float* pvalOfX, float alpha, int* candidates_c, size_t* yDim, size_t originalYDim,
                    bool splitted);

void perm_cuda_multi_all(const float* data_c,
              size_t data_height, size_t data_width, size_t permutations,
              float* x_permutations, int* sList_c,
              size_t sList_height, size_t sList_width, int x_id);

void perm_cuda_multi(const float*  obs,
               size_t           obs_count,
               size_t           dim,
               size_t           permutations,
               int * x_permutations);

size_t call_init_gpu_cuda();

int binom(int n, int k);