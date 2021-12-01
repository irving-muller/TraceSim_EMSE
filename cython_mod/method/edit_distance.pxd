# cython: language_level=3


# BASED ON

from cpython.version cimport PY_MAJOR_VERSION
from libc.stdlib cimport calloc, free
import numpy as np
cimport numpy as np

from cython_mod.method.method cimport Method

cdef class NeedlemanWunsch(Method):
    cdef double indel_penalty
    cdef double mismatch_penalty
    cdef double match_cost


    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil

cdef class DamerayLevenshtein(Method):
    cdef double insert_penalty
    cdef double delete_penalty
    cdef double subs_penalty
    cdef double trans_penalty
    cdef bint enable_trans

    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil