# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython_mod.util.structures cimport TermFreq
from libcpp.vector cimport vector

cdef class BowMethod:
    cdef double similarity(self, vector[TermFreq] & query, vector[TermFreq] & candidate) nogil:
        pass
