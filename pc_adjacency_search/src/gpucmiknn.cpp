// Python
#include <boost/python.hpp> 
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h> 
#include "gpucmiknn.h"
#include <iostream>

using namespace boost::python;

// CUDA pvalue wrapper
// Takes observations x and y and performs permutation-based independence test on them. It returns the pvalue.
float pval_l0(PyObject* data_, int k, size_t permutations)
{
  PyArrayObject* data = (PyArrayObject*) data_;
  size_t data_height = PyArray_DIM(data, 1);
  size_t data_width     = PyArray_DIM(data, 0);
  float* data_c = new float[data_height * data_width];

  for(size_t i = 0; i < data_height; i++) {
    for(size_t j = 0; j < data_width; j++) {
      data_c[data_height * j + i] =
          *(float*)PyArray_GETPTR2(data, j, i);
    }
  }
  
  float pval = 0.0;
  
  pval = pval_l0_cuda_shared(data_c, data_height, data_width, k, permutations);

  free(data_c);

  return pval;
}

float pval_ln(PyObject* data_, PyObject* x_permutations_, int k, int k_perm, size_t permutations)
{
  PyArrayObject* data = (PyArrayObject*) data_;
  size_t data_height = PyArray_DIM(data, 1);
  size_t data_width     = PyArray_DIM(data, 0);
  float* data_c = new float[data_height * data_width];

  for(size_t i = 0; i < data_height; i++) {
    for(size_t j = 0; j < data_width; j++) {
      data_c[data_height * j + i] =
         *(float*)PyArray_GETPTR2(data, j, i);
    }
  }

  PyArrayObject* x_permutations = (PyArrayObject*) x_permutations_;
  size_t x_permutations_height = PyArray_DIM(x_permutations, 1);
  size_t x_permutations_width     = PyArray_DIM(x_permutations, 0);
  float* x_permutations_c = new float[x_permutations_height * x_permutations_width];

  for(size_t i = 0; i < x_permutations_height; i++) {
    for(size_t j = 0; j < x_permutations_width; j++) {
      x_permutations_c[x_permutations_height * j + i] =
         *(float*)PyArray_GETPTR2(x_permutations, j, i);
    }
  }
  
  float pval = 0.0;
  
  pval = pval_ln_cuda(data_c, x_permutations_c, data_height, data_width, k, k_perm, permutations);

  free(data_c);
  free(x_permutations_c);

  return pval;
}

PyObject* pval_ln_row(PyObject* observations_, int x_id, PyObject* restricted_permutation_all_, int k, int k_perm, size_t permutations, int lvl, PyObject* sList_, float alpha, PyObject* candidates_, size_t originalYDim, bool splitted) {

  PyArrayObject* observations = (PyArrayObject*) observations_;
  size_t obs_count = PyArray_DIM(observations, 0);
  size_t vars     = PyArray_DIM(observations, 1);
  float* data_c = new float[obs_count * vars];
  for(size_t i = 0; i < vars; i++) {
    for(size_t j = 0; j < obs_count; j++) {
      data_c[obs_count * i + j] =
         *(float*)PyArray_GETPTR2(observations, j, i);
    }
  }

  PyArrayObject* sList = (PyArrayObject*) sList_;
  size_t dim = PyArray_DIM(sList, 1);
  size_t sEntries     = PyArray_DIM(sList, 0);
  int* sList_c = new int[dim * sEntries];

  for(size_t i = 0; i < sEntries; i++) {
    for(size_t j = 0; j < dim; j++) {
      sList_c[dim * i + j] =
         *(int*)PyArray_GETPTR2(sList, i, j);
    }
  }

  PyArrayObject* x_permutations_all = (PyArrayObject*) restricted_permutation_all_;
  float* x_permutations_c = new float[sEntries * permutations * obs_count];
  for(size_t i = 0; i < sEntries; i++) {
    for(size_t j = 0; j < permutations; j++) {
      for(size_t h = 0; h < obs_count; h++) {
      x_permutations_c[i * permutations * obs_count + j * obs_count + h] =
         *(float*)PyArray_GETPTR3(x_permutations_all, i, j, h);
      }
    }
  }

  PyArrayObject* ys = (PyArrayObject*) candidates_;
  size_t yDim     = PyArray_DIM(ys, 0);
  int* candidates_c = new int[yDim];

  for(size_t i = 0; i < yDim; i++) {
    candidates_c[i] = *(int*)PyArray_GETPTR1(ys, i);
  }

  int* sOfX = new int[vars * lvl];
  memset(sOfX, -1, vars * lvl * sizeof(int));
  float* pvalOfX = new float[vars];
  memset(pvalOfX, 0, vars * sizeof(float));
  pval_ln_row_cuda(data_c, x_permutations_c, obs_count, k, k_perm, permutations, x_id, vars, lvl, sList_c, sEntries, sOfX, pvalOfX, alpha, candidates_c, &yDim, originalYDim, splitted);

  npy_intp dims[2] = {(npy_intp)vars, (npy_intp)lvl};
  PyArrayObject* sOfX_ = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_INT, sOfX);
  npy_intp dimsP[1] = {(npy_intp)vars};
  PyArrayObject* pvalOfX_ = (PyArrayObject *) PyArray_SimpleNewFromData(1, dimsP, NPY_FLOAT, pvalOfX);
  npy_intp dimsY[1] = {(npy_intp)yDim};
  PyArrayObject* candidatesY_ = (PyArrayObject *) PyArray_SimpleNewFromData(1, dimsY, NPY_INT, candidates_c);
  PyArray_ENABLEFLAGS(sOfX_, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS(pvalOfX_, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS(candidatesY_, NPY_ARRAY_OWNDATA);
  free(data_c);
  free(x_permutations_c);
  free(sList_c);
  return PyTuple_Pack(3, sOfX_, pvalOfX_, candidatesY_);
}


static PyObject *pval_l0_row(PyObject* observations_, int x_id, int k, size_t permutations, PyObject* candidates_) {

  PyArrayObject* observations = (PyArrayObject*) observations_;
  size_t obs_count = PyArray_DIM(observations, 0);
  size_t vars     = PyArray_DIM(observations, 1);
  float* data_c = new float[obs_count * vars];
  for(size_t i = 0; i < vars; i++) {
    for(size_t j = 0; j < obs_count; j++) {
      data_c[obs_count * i + j] =
         *(float*)PyArray_GETPTR2(observations, j, i);
    }
  }

  PyArrayObject* ys = (PyArrayObject*) candidates_;
  size_t yDim     = PyArray_DIM(ys, 0);
  int* candidates_c = new int[yDim];

  for(size_t i = 0; i < yDim; i++) {
    candidates_c[i] = *(int*)PyArray_GETPTR1(ys, i);
  }

  float* pvalOfX = new float[yDim];
  memset(pvalOfX, 0, yDim * sizeof(float));
  pval_l0_row_cuda(data_c, obs_count, k, permutations, x_id, vars, pvalOfX, candidates_c, yDim);

  npy_intp dimsP[1] = {(npy_intp)yDim};
  PyArrayObject* pvalOfX_ = (PyArrayObject *) PyArray_SimpleNewFromData(1, dimsP, NPY_FLOAT, pvalOfX);
  PyArray_ENABLEFLAGS(pvalOfX_, NPY_ARRAY_OWNDATA);
  free(data_c);
  free(candidates_c);

  return PyArray_Return(pvalOfX_);
}

static PyObject *rperm_multi_all(PyObject* observations_, PyObject* sList_, size_t permutations, int x_id) {
  PyArrayObject* observations = (PyArrayObject*) observations_;
  size_t obs_count = PyArray_DIM(observations, 0);
  size_t vars     = PyArray_DIM(observations, 1);
  float* data_c = new float[obs_count * vars];
  for(size_t i = 0; i < vars; i++) {
    for(size_t j = 0; j < obs_count; j++) {
      data_c[obs_count * i + j] =
         *(float*)PyArray_GETPTR2(observations, j, i);
    }
  }

  PyArrayObject* sList = (PyArrayObject*) sList_;
  size_t dim = PyArray_DIM(sList, 1);
  size_t sEntries     = PyArray_DIM(sList, 0);
  int* sList_c = new int[dim * sEntries];
  for(size_t i = 0; i < sEntries; i++) {
    for(size_t j = 0; j < dim; j++) {
      sList_c[dim * i + j] =
         *(int*)PyArray_GETPTR2(sList, i, j);
    }
  }
  float* x_permutations = new float[permutations * obs_count * sEntries];

  perm_cuda_multi_all(data_c, obs_count, vars, permutations, x_permutations, sList_c, dim, sEntries, x_id);

  npy_intp dims[3] = {(npy_intp)sEntries, (npy_intp)permutations, (npy_intp)obs_count};
  PyArrayObject* py_obj_dist = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, x_permutations);
  PyArray_ENABLEFLAGS(py_obj_dist, NPY_ARRAY_OWNDATA);
  
  free(data_c);
  free(sList_c);
  
  return PyArray_Return(py_obj_dist);

}

static PyObject *rperm_multi(PyObject* z_, size_t permutations)
{
  PyArrayObject* z = (PyArrayObject*) z_;
  size_t data_height = PyArray_DIM(z, 1);
  size_t data_width     = PyArray_DIM(z, 0);
  float* data_c = new float[data_height * data_width];

  for(size_t i = 0; i < data_height; i++) {
    for(size_t j = 0; j < data_width; j++) {
      data_c[data_height * j + i] =
         *(float*)PyArray_GETPTR2(z, j, i);
    }
  }

  int* x_permutations = new int[permutations * data_height];
    
  perm_cuda_multi(data_c, data_height, data_width, permutations, x_permutations);

  npy_intp dims[2] = {(npy_intp)permutations, (npy_intp)data_height};
  PyArrayObject* py_obj_dist = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_INT, x_permutations);
  PyArray_ENABLEFLAGS(py_obj_dist, NPY_ARRAY_OWNDATA);
  
  free(data_c);
  return PyArray_Return(py_obj_dist);
}

size_t init_gpu()
{
  return call_init_gpu_cuda();
}

BOOST_PYTHON_MODULE(gpucmiknn)
{
  import_array();
  def("pval_l0", pval_l0);
  def("pval_ln", pval_ln);
  def("rperm_multi", rperm_multi);
  def("rperm_multi_all", rperm_multi_all);
  def("pval_ln_row", pval_ln_row);
  def("init_gpu", init_gpu);
  def("pval_l0_row", pval_l0_row);
}

