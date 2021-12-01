# cython: language_level=3

"""
Position Dependent Model
"""

from libc.math cimport exp
from libc.stdlib cimport calloc, free

from cython_mod.method.method cimport Method

import numpy
cimport numpy


cdef class PDM(Method):
    cdef double c
    cdef double o

    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil
