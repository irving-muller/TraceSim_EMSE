# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython_mod.method.method cimport Method, DTYPE_t

import numpy
cimport numpy


cdef struct Frame:
    int id
    double weight

cdef class TraceSim(Method):
    cdef DTYPE_t[:] df
    cdef double df_coef
    cdef double pos_coef
    cdef double diff_coef
    cdef double match_cost
    cdef double gap_penalty
    cdef double mismatch_penalty

    cdef bint sigmoid
    cdef double gamma
    cdef bint sum
    cdef bint idf
    cdef bint const_match
    cdef bint reciprocal_func
    cdef bint no_norm
    cdef bint const_gap
    cdef bint const_mismatch
    cdef bint brodie_function

    cdef double calculate_weight(self, int * trace, Py_ssize_t pos, long seq_len) nogil
    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil
