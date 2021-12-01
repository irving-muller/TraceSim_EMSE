# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython_mod.method.method cimport Method


cdef class PrefixMatch(Method):
    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil

