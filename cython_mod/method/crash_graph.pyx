# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython_mod.method.method cimport Method

cdef class CrashGraph(Method):
    cdef double similarity(self, int *query, int *candidate, long query_len, long cand_len) nogil:
        cdef Py_ssize_t i = 0, j = 0
        cdef double n_edges = 0
        while i < query_len and j < cand_len:
            if query[i] == candidate[j]:
                i += 1
                j += 1
                n_edges += 1.0
            elif query[i] > candidate[j]:
                j += 1
            else:
                i += 1

        return n_edges / min(query_len, cand_len)
