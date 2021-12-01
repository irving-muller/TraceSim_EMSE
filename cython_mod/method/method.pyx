# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp


cdef class Method:
    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil:
        pass
