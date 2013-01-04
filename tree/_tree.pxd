# Author: Peter Prettenhofer, Brian Holt, Gilles Louppe
# License: BSD Style.

# See _tree.pyx for details.

cimport numpy as np

ctypedef np.float32_t DTYPE_t
ctypedef np.float64_t DOUBLE_t
ctypedef np.int8_t BOOL_t


# ==============================================================================
# Criterion
# ==============================================================================

cdef class Criterion:
    # Methods
    cdef void init(self, DOUBLE_t* y, int y_stride, BOOL_t* sample_mask,
                   int n_samples, int n_total_samples)

    cdef void reset(self)

    cdef int update(self, int a, int b, DOUBLE_t* y, int y_stride,
                    int* X_argsorted_i, BOOL_t* sample_mask)

    cdef double eval(self)

    cdef void init_value(self, double* buffer_value)


# ==============================================================================
# Tree
# ==============================================================================

cdef class Tree:
    # Input/Output layout
    cdef public int n_features
    cdef int* n_classes
    cdef public int n_outputs

    cdef public int max_n_classes
    cdef public int value_stride

    # Parameters
    cdef public Criterion criterion
    cdef public double max_depth
    cdef public int min_samples_split
    cdef public int min_samples_leaf
    cdef public double min_density
    cdef public int max_features
    cdef public int find_split_algorithm
    cdef public object random_state

    # Inner structures
    cdef public int node_count
    cdef public int capacity
    cdef int* children_left
    cdef int* children_right
    cdef int* feature
    cdef double* threshold
    cdef double* value
    cdef double* best_error
    cdef double* init_error
    cdef int* n_samples

    # Methods
    cdef void resize(self, int capacity=*)

    cpdef build(self, np.ndarray X, np.ndarray y,
                np.ndarray sample_mask=*, np.ndarray X_argsorted=*)

    cdef void recursive_partition(self,
                                  np.ndarray[DTYPE_t, ndim=2, mode="fortran"] X,
                                  np.ndarray[np.int32_t, ndim=2, mode="fortran"] X_argsorted,
                                  np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                                  np.ndarray sample_mask,
                                  int n_node_samples,
                                  int depth,
                                  int parent,
                                  int is_left_child,
                                  double* buffer_value)

    cdef int add_split_node(self, int parent, int is_left_child, int feature,
                                  double threshold, double* value,
                                  double best_error, double init_error,
                                  int n_samples)

    cdef int add_leaf(self, int parent, int is_left_child, double* value,
                      double error, int n_samples)

    cdef void find_split(self, DTYPE_t* X_ptr, int X_stride,
                         int* X_argsorted_ptr, int X_argsorted_stride,
                         DOUBLE_t* y_ptr, int y_stride, BOOL_t* sample_mask_ptr,
                         int n_node_samples, int n_total_samples, int* _best_i,
                         double* _best_t, double* _best_error,
                         double* _initial_error)

    cdef void find_best_split(self, DTYPE_t* X_ptr, int X_stride,
                              int* X_argsorted_ptr, int X_argsorted_stride,
                              DOUBLE_t* y_ptr, int y_stride,
                              BOOL_t* sample_mask_ptr, int n_node_samples,
                              int n_total_samples, int* _best_i,
                              double* _best_t, double* _best_error,
                              double* _initial_error)

    cdef void find_random_split(self, DTYPE_t* X_ptr, int X_stride,
                                int* X_argsorted_ptr, int X_argsorted_stride,
                                DOUBLE_t* y_ptr, int y_stride,
                                BOOL_t* sample_mask_ptr, int n_node_samples,
                                int n_total_samples, int* _best_i,
                                double* _best_t, double* _best_error,
                                double* _initial_error)

    cpdef predict(self, np.ndarray[DTYPE_t, ndim=2] X)

    cpdef apply(self, np.ndarray[DTYPE_t, ndim=2] X)

    cpdef compute_feature_importances(self, method=*)

    cdef inline double _compute_feature_importances_gini(self, int node)

    cdef inline double _compute_feature_importances_squared(self, int node)
