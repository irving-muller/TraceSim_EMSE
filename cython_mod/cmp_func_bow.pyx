
# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp


from cython_mod.method.method cimport DTYPE_t
from cython_mod.bow_method.bow_method cimport BowMethod, BowSimilarityMethod, DTYPE_t
from cython_mod.util.structures cimport TermFreq
from cython_mod.bow_method.durfex cimport Durfex

from libcpp.vector cimport vector

from cython.parallel import prange
from libc.stdlib cimport malloc, free

from libc.stdio cimport printf

import numpy
cimport numpy


cdef to_bow(vector[TermFreq] & stacks, int report_idx, vector[int] report_stack):
    cdef Py_ssize_t idx
    cdef int * stack_c
    cdef int function_id
    cdef TermFreq tf
    cdef int previous_fun = report_stack[0]
    cdef double freq = 1.0

    for idx in range(1, report_stack.size()):
        function_id = report_stack[idx]

        if previous_fun < function_id:
            tf = TermFreq()

            tf.term = previous_fun
            tf.freq = freq

            stacks.push_back(tf)
            freq = 1.0
            previous_fun = function_id
        elif previous_fun == function_id:
            freq+=1.0
        else:
            raise Exception("Stack trace has to be sorted.")

    if freq != 0:
        tf.term = previous_fun
        tf.freq = freq

        stacks.push_back(tf)

cdef to_bow_by_report(vector[vector[TermFreq]] & stacks_by_reportid, report_stacks):
    cdef Py_ssize_t report_idx
    cdef vector[TermFreq] stacks
    cdef vector[int] st

    for report_idx in range(len(report_stacks)):
        stacks = vector[TermFreq]()
        st = report_stacks[report_idx][0]

        stacks.reserve(st.size())

        to_bow(stacks, report_idx, st)
        stacks_by_reportid.push_back(stacks)


cpdef compare_bow(int n_threads, BowSimilarityMethod method_type, args, query, candidates, numpy.ndarray[DTYPE_t, ndim=1] idf):
    cdef vector[TermFreq] query_stack
    cdef vector[vector[TermFreq]] stacks_by_report
    cdef Py_ssize_t n_candidates = len(candidates)

    if len(query) > 1:
        raise Exception("Query length should be 1")

    cdef vector[int] query_list = query[0]

    query_stack.reserve(query_list.size())
    stacks_by_report.reserve(n_candidates)

    # Copy stacktraces
    to_bow(query_stack, 0, query_list)
    to_bow_by_report(stacks_by_report, candidates)

    cdef Py_ssize_t report_idx, cand_stack_idx
    cdef BowMethod method

    if method_type == BowSimilarityMethod.DURFEX:
        # DTYPE_t[:] df, double df_coef, double pos_coef, double diff_coef, double match_cost, double gap_penalty, double mismatch_penalty
        method = Durfex(idf, *args)

    # for cand_stack_idx in range(n_cand_stacks):
    cdef numpy.ndarray[DTYPE_t, ndim=1] results_np = numpy.ones([n_candidates], dtype=numpy.double) * -9999999.99
    cdef DTYPE_t[:] results = results_np

    cdef Py_ssize_t i, j
    cdef double result, score

    for report_idx in prange(n_candidates, num_threads=n_threads, nogil=True):
        if stacks_by_report[report_idx].size() == 0:
            continue

        results[report_idx] = method.similarity(query_stack, stacks_by_report[report_idx])

    query_stack.clear()
    stacks_by_report.clear()


    return results_np.tolist()
