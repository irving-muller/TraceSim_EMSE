# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython_mod.method.method cimport Method, DTYPE_t

import numpy
cimport numpy


cdef class Brodie05(Method):
    cdef DTYPE_t[:] idf
    cdef double coef_gap
    cdef double gap_penalty
    cdef double mismatch_penalty

    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil
