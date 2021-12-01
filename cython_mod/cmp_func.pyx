
# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp


from cython_mod.method.method cimport Method, SimilarityMethod, DTYPE_t
from cython_mod.method.brodie_05 cimport Brodie05
from cython_mod.method.edit_distance cimport DamerayLevenshtein, NeedlemanWunsch
from cython_mod.method.trace_sim cimport TraceSim
from cython_mod.method.crash_graph cimport CrashGraph
from cython_mod.method.pdm cimport PDM
from cython_mod.method.prefix_match cimport PrefixMatch
from cython_mod.util.cython_util cimport clean, to_c, to_c_by_report
from cython_mod.util.structures cimport Stacktrace
from cython_mod.util.comparator cimport Comparator, Max, AggStrategy, Mean, MeanType, WeightType, StackTraceInfo, Alignment, FilterStrategy, Filter, SelectOne, KTopFunction

from libcpp.vector cimport vector
from libc.stdio cimport printf

from cython.parallel import prange
from libc.stdlib cimport malloc, free

import numpy
cimport numpy

cpdef compare(int n_threads, SimilarityMethod method_type, AggStrategy agg_strategy, args, query, candidates, numpy.ndarray[DTYPE_t, ndim=1] df, FilterStrategy filter_strategy, double filter_k,  numpy.ndarray[numpy.uint8_t, ndim=1] is_stop_word, bint beg_trail_trim):
    cdef vector[Stacktrace] query_stacks
    cdef vector[vector[Stacktrace]] stacks_by_report

    query_stacks.reserve(len(query))
    stacks_by_report.reserve(3 * len(candidates))

    # Copy stacktraces
    to_c(query_stacks, 0, query, is_stop_word, beg_trail_trim)
    to_c_by_report(stacks_by_report, candidates, is_stop_word, beg_trail_trim)

    cdef Py_ssize_t report_idx, cand_stack_idx
    cdef DTYPE_t[:] df_array = df

    cdef Comparator agg
    cdef Py_ssize_t n_candidates = len(candidates)
    cdef Method method
    cdef numpy.ndarray[DTYPE_t, ndim=1] d

    if method_type == SimilarityMethod.TRACE_SIM:
        # DTYPE_t[:] df, double df_coef, double pos_coef, double diff_coef, double match_cost, double gap_penalty, double mismatch_penalty
        method = TraceSim(df, *args)
    elif method_type == SimilarityMethod.BRODIE_05:
        method = Brodie05(1.0 - (df/100.00), *args)
    elif method_type == SimilarityMethod.DAMERAU_LEVENSHTEIN:
        method = DamerayLevenshtein(*args)
    elif method_type == SimilarityMethod.OPT_ALIGN:
        method = NeedlemanWunsch(*args)
    elif method_type == SimilarityMethod.PDM_METHOD:
        method = PDM(*args)
    elif method_type == SimilarityMethod.PREFIX_MATCH:
        method = PrefixMatch()
    elif method_type == SimilarityMethod.CRASH_GRAPH:
        method = CrashGraph()

    cdef Filter filter

    if filter_strategy == FilterStrategy.SELECT_ONE:
        filter = SelectOne(df_array)
    elif filter_strategy == FilterStrategy.TOP_K_FUNC:
        filter = KTopFunction(filter_k, df_array)
    elif filter_strategy == FilterStrategy.NONE:
        filter = Filter()


    cdef MeanType mean_type
    cdef WeightType weight_type

    if agg_strategy == AggStrategy.MAX:
        agg = Max()
        # print("MAX")
    else:
        if agg_strategy == AggStrategy.AVG_QUERY:
            mean_type = MeanType.QUERY
            weight_type = WeightType.OFF
        elif agg_strategy == AggStrategy.AVG_CAND:
            mean_type = MeanType.CANDIDATE
            weight_type = WeightType.OFF
        elif agg_strategy == AggStrategy.AVG_SHORT:
            mean_type = MeanType.SHORTEST
            weight_type = WeightType.OFF
        elif agg_strategy == AggStrategy.AVG_LONG:
            mean_type = MeanType.LONGEST
            weight_type = WeightType.OFF
        elif agg_strategy == AggStrategy.AVG_QUERY_CAND:
            mean_type = MeanType.QUERY_CAND
            weight_type = WeightType.OFF

        # print("Mean")
        agg = Mean(mean_type, weight_type,  df_array)


    # for cand_stack_idx in range(n_cand_stacks):
    cdef numpy.ndarray[DTYPE_t, ndim=1] results_np = numpy.ones([n_candidates], dtype=numpy.double) * -9999999.99
    cdef DTYPE_t[:] results = results_np

    # for report_idx in range(n_candidates):
    cdef vector[Stacktrace] f_query_stacks
    cdef vector[Stacktrace] f_cand_stacks
    cdef vector[StackTraceInfo] query_stacks_info
    cdef vector[StackTraceInfo] cand_stacks_info
    cdef Stacktrace * cand_st
    cdef Stacktrace * query_st
    cdef Py_ssize_t i, j
    cdef double result, score
    cdef double * matrix_score

    f_query_stacks = filter.filter(query_stacks)

    if f_query_stacks.size() > 0:
        for report_idx in prange(n_candidates, num_threads=n_threads, nogil=True):
        # for report_idx in range(n_candidates):
        #     printf("\n\n%d\n", report_idx)
        #     printf("Report %d size=%d\n", report_idx, stacks_by_report[report_idx].size())
            if stacks_by_report[report_idx].size() == 0:
                # printf("\tEmpty %d\n", report_idx)
                continue

            f_cand_stacks = filter.filter(stacks_by_report[report_idx])

            if f_cand_stacks.size() == 0:
                continue


            query_stacks_info = agg.prepare(f_query_stacks, stacks_by_report[report_idx])
            cand_stacks_info = agg.prepare(f_cand_stacks, query_stacks)

            matrix_score = <double *> malloc(query_stacks_info.size() * cand_stacks_info.size() *  sizeof(double))

            # printf("\t%d,%d\n",query_stacks_info.size(), cand_stacks_info.size())
            # printf("------------------------------\n")
            for i in range(query_stacks_info.size()):
                query_st = query_stacks_info[i].stack
                for j in range(cand_stacks_info.size()):
                    cand_st = cand_stacks_info[j].stack

                    # score = 0
                    score =  method.similarity(query_st.stack, cand_st.stack, query_st.length, cand_st.length)

                    matrix_score[(i * cand_stacks_info.size()) + j] = score
                    # printf("%d\t%d\t%d\t%f\n", cand_st.report_idx, i, j, matrix_score[(i * cand_stacks_info.size()) + j])
                # printf("\n")

            result = agg.aggregate(matrix_score, query_stacks_info, cand_stacks_info)
            free(matrix_score)

            results[report_idx] =  result

    clean(query_stacks, stacks_by_report)

    return results_np.tolist()
