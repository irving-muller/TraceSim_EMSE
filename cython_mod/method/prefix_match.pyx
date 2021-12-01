# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from libc.stdlib cimport calloc, free
from libcpp.vector cimport vector
from libc.math cimport exp, fabs

from cython.parallel import prange

import numpy
cimport numpy

cdef Py_ssize_t ONE_AGO = 0
cdef Py_ssize_t THIS_ROW = 1

from libc.stdio cimport printf

###################################################
# Brodie 05
###################################################

cdef class PrefixMatch(Method):
    cdef double similarity(self, int *query, int *candidate, long query_len, long cand_len) nogil:
        cdef double lcp = 0

        cdef Py_ssize_t i
        cdef long min_len = min(query_len, cand_len)

        for i in range(min(query_len, cand_len)):
            if query[i] != candidate[i]:
                break

            lcp += 1.0

        return lcp / max(query_len, cand_len)
