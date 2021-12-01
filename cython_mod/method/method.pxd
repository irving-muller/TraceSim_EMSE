# distutils: language = c++
# cython: language_level=3

import numpy
cimport numpy

ctypedef numpy.double_t DTYPE_t

cpdef enum SimilarityMethod:
    TRACE_SIM=0
    BRODIE_05=1
    DAMERAU_LEVENSHTEIN=2
    OPT_ALIGN=3
    PDM_METHOD = 4
    PREFIX_MATCH = 5
    CRASH_GRAPH=6

cdef class Method:
    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil
