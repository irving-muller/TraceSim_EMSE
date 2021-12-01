# distutils: language = c++
# cython: language_level=3

import numpy
cimport numpy

from cython_mod.method.method cimport DTYPE_t
from cython_mod.util.structures cimport TermFreq
from libcpp.vector cimport vector

cpdef enum BowSimilarityMethod:
    DURFEX = 7

cdef class BowMethod:
    cdef double similarity(self, vector[TermFreq] & query, vector[TermFreq] & candidate) nogil
