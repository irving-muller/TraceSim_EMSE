# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from libc.stdlib cimport calloc, free
from libcpp.algorithm cimport sort
from libc.math cimport sqrt
from libc.stdio cimport printf
import numpy
cimport numpy
from cython_mod.util.structures cimport TermFreq
from libcpp.vector cimport vector

from libc.stdio cimport printf

cimport cython

cdef double get_stack_trace_size(vector[TermFreq] & st) nogil:
    cdef double n_call_query = 0.0

    for s in st:
        n_call_query += s.freq

    return n_call_query

cdef class Durfex(BowMethod):
    def __cinit__(self, numpy.ndarray[DTYPE_t, ndim=1] idf):
        # self.idf = numpy.log(idf)
        self.idf = idf

    cdef double similarity(self, vector[TermFreq] & query, vector[TermFreq] & candidate) nogil:
        cdef Py_ssize_t q_idx = 0
        cdef Py_ssize_t c_idx = 0
        cdef TermFreq *q_ptr
        cdef TermFreq *c_ptr
        cdef Py_ssize_t query_size = query.size()
        cdef Py_ssize_t cand_size = candidate.size()
        cdef double sum = 0.0
        cdef double squared_sum_query = 0.0
        cdef double squared_sum_cand = 0.0

        cdef double query_weight
        cdef double cand_weight

        cdef TermFreq void_obj

        void_obj.term = max(query.back().term, candidate.back().term) + 1

        cdef TermFreq *null_obj_ptr = &void_obj
        cdef double q_IDF
        cdef double c_IDF

        cdef double st_size_query = get_stack_trace_size(query)
        cdef double st_size_cand = get_stack_trace_size(candidate)

        while True:
            if q_idx < query_size:
                q_ptr = &query.at(q_idx)
                query_weight = (q_ptr.freq / st_size_query) * (self.idf[q_ptr.term])
            else:
                q_ptr = null_obj_ptr
                query_weight = 0

            if c_idx < cand_size:
                c_ptr = &candidate.at(c_idx)
                cand_weight = (c_ptr.freq / st_size_cand) * (self.idf[c_ptr.term])
            else:
                c_ptr = null_obj_ptr
                cand_weight = 0.0

            if q_ptr == null_obj_ptr and c_ptr == null_obj_ptr:
                break

            # printf("New It\n")
            if q_ptr.term < c_ptr.term:
                squared_sum_query += query_weight * query_weight
                q_idx += 1
            elif c_ptr.term < q_ptr.term:
                # printf("Cand\n")
                squared_sum_cand += cand_weight * cand_weight
                c_idx += 1
            else:
                sum += cand_weight * query_weight

                squared_sum_query += query_weight * query_weight
                squared_sum_cand += cand_weight * cand_weight

                q_idx += 1
                c_idx += 1

        if sum == 0.0:
            return 0.0

        return sum / (sqrt(squared_sum_query) * sqrt(squared_sum_cand))
