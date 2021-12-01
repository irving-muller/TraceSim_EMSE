# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython_mod.bow_method.bow_method cimport BowMethod, DTYPE_t
from cython_mod.util.structures cimport TermFreq
from libcpp.vector cimport vector


cdef class Durfex(BowMethod):
    cdef DTYPE_t[:] idf

    cdef double similarity(self, vector[TermFreq] & query, vector[TermFreq] & candidate) nogil
