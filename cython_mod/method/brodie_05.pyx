# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from libc.stdlib cimport calloc, free
from libcpp.vector cimport vector
from libc.math cimport exp, fabs

from cython.parallel import prange


import numpy
cimport numpy

cdef Py_ssize_t ONE_AGO = 0
cdef Py_ssize_t THIS_ROW = 1

from libc.stdio cimport printf

###################################################
# Brodie 05
###################################################

cdef class Brodie05(Method):
    def __cinit__(self, DTYPE_t[:] &idf, double coef_gap, double gap_penalty, double mismatch_penalty):
        self.idf = idf
        self.coef_gap = coef_gap
        self.gap_penalty = gap_penalty
        self.mismatch_penalty = mismatch_penalty

    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil:
        cdef Py_ssize_t q_idx, c_idx
        # We need to keep only two rows of the matrix M

        cdef Py_ssize_t offset = cand_len + 1
        cdef double *M = <double *> calloc(2 * offset, sizeof(double))
        cdef double previous_row, previous_col, previous_row_col, cost, sim, normalized_sim
        cdef double query_len_double = query_len
        cdef int q_pos, c_pos
        cdef Py_ssize_t i, j, k
        cdef double min_value, max_value

        # Normalize output
        cdef double *WS = <double *> calloc(2 * offset, sizeof(double))

        try:
            # Create first row
            for i in range(offset):
                M[THIS_ROW * offset + i] = -self.gap_penalty * i
                WS[THIS_ROW * offset + i] = -self.gap_penalty * i

            for i in range(query_len):
                q_pos = i
                i += 1

                # Copy THIS_ROW to ONE_AGO
                for k in range(offset):
                    M[ONE_AGO * offset + k] = M[THIS_ROW * offset + k]
                    WS[ONE_AGO * offset + k] = WS[THIS_ROW * offset + k]

                # Reset THIS_ROW
                for k in range(offset):
                    M[THIS_ROW * offset + k] = 0.0
                    WS[THIS_ROW * offset + k] = 0.0

                # Set first column of the row
                M[THIS_ROW * offset + 0] = -self.gap_penalty * i
                WS[THIS_ROW * offset + 0] = -self.gap_penalty * i

                for j in range(cand_len):
                    c_pos = j
                    j += 1

                    # Gap
                    previous_row = M[ONE_AGO * offset + j] - self.gap_penalty
                    previous_col = M[THIS_ROW * offset + j - 1] - self.gap_penalty
                    previous_row_col = M[ONE_AGO * offset + j - 1]

                    if query[q_pos] == candidate[c_pos]:
                        # IDF * Function call position * Shift between calls
                        cost = self.idf[query[q_pos]] * (1.0 - (q_pos / query_len_double)) * exp(
                            -self.coef_gap * fabs(q_pos - c_pos) / 2.0)
                        # print("\t{}={}+{}: {} * {} * {} ".format(
                        #     previous_row_col + cost,
                        #     previous_row_col,
                        #     cost,
                        #     idf[query[q_pos]], (1.0 - (q_pos / query_len_double)), exp(-coef_gap * abs(q_pos - c_pos)/ 2.0)))

                        previous_row_col += cost
                    else:
                        previous_row_col -= self.mismatch_penalty

                    M[THIS_ROW * offset + j] = max(previous_row, previous_col, previous_row_col)

                    WS[THIS_ROW * offset + j] = max(WS[ONE_AGO * offset + j] - self.gap_penalty,
                                                    WS[THIS_ROW * offset + j - 1] - self.gap_penalty,
                                                    WS[ONE_AGO * offset + j - 1] - self.mismatch_penalty)

                    # print("i={} j={} ({},{},{}) {}".format(i, j, previous_row, previous_row_col, previous_col,
                    #                                        M[THIS_ROW * offset + j]))
                # print("")

            sim = M[THIS_ROW * offset + offset - 1]

            # Min values
            min_value = WS[THIS_ROW * offset + offset - 1]
            max_value = 0

            for i in range(query_len):
                # print("{}\t{} * {}".format(idf[query[i]] * (1.0 - (i / query_len_double)),
                #                                                                               idf[query[i]], 1.0 - (i / query_len_double)))
                max_value += self.idf[query[i]] * (1.0 - (i / query_len_double))

            # printf("SIM: %f\tmin=%f , max=%f\t Normalized sim: %f\n",sim, min_value, max_value, 0)
            # normalized_sim = (sim - min_value) / (max_value - min_value)
            normalized_sim = sim

        finally:
            free(M)
            free(WS)

        return normalized_sim


