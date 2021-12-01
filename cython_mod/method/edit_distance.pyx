# cython: language_level=3


# BASED ON

from cpython.version cimport PY_MAJOR_VERSION
from libc.stdlib cimport calloc, free
import numpy as np
cimport numpy as np
from libc.stdio cimport printf

from cython_mod.method.method cimport Method

cdef class NeedlemanWunsch(Method):
    def __cinit__(self, double indel_penalty, double mismatch_penalty, double match_cost):
        self.indel_penalty = indel_penalty
        self.mismatch_penalty = mismatch_penalty
        self.match_cost = match_cost

    cdef double similarity(self, int *query, int *candidate, long query_len, long cand_len) nogil:
        #Index
        cdef Py_ssize_t ONE_AGO = 0
        cdef Py_ssize_t THIS_ROW = 1

        cdef Py_ssize_t q_idx, c_idx
        # We need to keep only two rows of the matrix M

        cdef Py_ssize_t offset = cand_len + 1
        cdef double *M = <double *> calloc(2 * offset, sizeof(double))
        cdef double *WS = <double *> calloc(2 * offset, sizeof(double))

        # print("q_len: {}\tc_len: {}".format(query_len, cand_len))
        # printf("indel_penalty=%f, mismatch_penalty=%f, match_cost=%f\n", self.indel_penalty, self.mismatch_penalty,
        #       self.match_cost)

        cdef double previous_row, previous_col, previous_row_col, sim, normalized_sim
        cdef int q_pos, c_pos
        cdef Py_ssize_t i, j, k
        cdef double min_value, max_value

        try:
            # Create first row
            for i in range(offset):
                M[THIS_ROW * offset + i] = -self.indel_penalty * i
                WS[THIS_ROW * offset + i] = -self.indel_penalty * i

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
                M[THIS_ROW * offset + 0] = -self.indel_penalty * i
                WS[THIS_ROW * offset + 0] = -self.indel_penalty * i

                for j in range(cand_len):
                    c_pos = j
                    j += 1

                    # Gap
                    previous_row = M[ONE_AGO * offset + j] - self.indel_penalty
                    previous_col = M[THIS_ROW * offset + j - 1] - self.indel_penalty
                    previous_row_col = M[ONE_AGO * offset + j - 1]

                    if query[q_pos] == candidate[c_pos]:
                        previous_row_col += self.match_cost
                    else:
                        previous_row_col -= self.mismatch_penalty

                    M[THIS_ROW * offset + j] = max(previous_row, previous_col, previous_row_col)
                    WS[THIS_ROW * offset + j] = max(WS[ONE_AGO * offset + j] - self.indel_penalty,
                                                    WS[THIS_ROW * offset + j - 1] - self.indel_penalty,
                                                    WS[ONE_AGO * offset + j - 1] - self.mismatch_penalty)

                    # print("i={} j={} ({},{},{}) {}".format(i, j, previous_row, previous_row_col, previous_col,
                    #                                        M[THIS_ROW * offset + j]))
                # print("")

            sim = M[THIS_ROW * offset + offset - 1]

            # Min value
            min_value = WS[THIS_ROW * offset + offset - 1]
            max_value = min(query_len, cand_len) * self.match_cost

            normalized_sim = (sim - min_value) / (max_value - min_value)

            # printf("SIM: %f\tmin=%f , max=%f\t Normalized sim: %f",sim, min_value, max_value, normalized_sim)

            return normalized_sim
        finally:
            free(M)
            free(WS)

cdef class DamerayLevenshtein(Method):
    def __cinit__(self, double insert_penalty, double delete_penalty, double subs_penalty, double trans_penalty,
                  bint enable_trans):
        self.insert_penalty = insert_penalty
        self.delete_penalty = delete_penalty
        self.subs_penalty = subs_penalty
        self.trans_penalty = trans_penalty
        self.enable_trans = enable_trans

    cdef double similarity(self, int *query, int *candidate, long query_len, long cand_len) nogil:
        """
            https://pypi.org/project/pyxDamerauLevenshtein/
            
            >>> damerau_levenshtein_distance('smtih', 'smith')
            1
            >>> damerau_levenshtein_distance('saturday', 'sunday')
            3
            >>> damerau_levenshtein_distance('orange', 'pumpkin')
            7
        """
        # index
        cdef Py_ssize_t TWO_AGO = 0
        cdef Py_ssize_t ONE_AGO = 1
        cdef Py_ssize_t THIS_ROW = 2

        # printf(
        # "insert_penalty=%f, delete_penalty=%f, subs_penalty=%f, trans_penalty=%f, enable_trans=%d\n", self.insert_penalty,
        # self.delete_penalty, self.subs_penalty, self.trans_penalty, self.enable_trans)

        # possible short-circuit if words have a lot in common at the beginning (or are identical)
        # cdef Py_ssize_t first_differing_index = 0
        # while first_differing_index < len(query) and \
        #         first_differing_index < len(candidate) and \
        #         query[first_differing_index] == candidate[first_differing_index]:
        #     first_differing_index += 1
        #
        # query = query[first_differing_index:]
        # candidate = candidate[first_differing_index:]

        # Py_ssize_t should be used wherever we're dealing with an array index or length
        cdef Py_ssize_t i, j
        cdef Py_ssize_t offset = cand_len + 1
        cdef float delete_cost, add_cost, subtract_cost, edit_distance

        # storage is a 3 x (len(candidate) + 1) array that stores TWO_AGO, ONE_AGO, and THIS_ROW
        cdef float *M = <float *> calloc(3 * offset, sizeof(float))
        cdef float *WS = <float *> calloc(2 * offset, sizeof(float))

        cdef Py_ssize_t THIS_ROW_WS = 1
        cdef Py_ssize_t ONE_AGO_WS = 0

        try:
            # initialize THIS_ROW
            for i in range(1, offset):
                # Insert from query to candidate
                M[THIS_ROW * offset + (i - 1)] = i * self.insert_penalty
                WS[THIS_ROW_WS * offset + (i - 1)] = i * self.insert_penalty

            # print("I: {} {}".format(query_len, s2_len))
            for i in range(query_len):
                # swap/initialize vectors
                for j in range(offset):
                    M[TWO_AGO * offset + j] = M[ONE_AGO * offset + j]
                    M[ONE_AGO * offset + j] = M[THIS_ROW * offset + j]

                    WS[ONE_AGO_WS * offset + j] = WS[THIS_ROW_WS * offset + j]

                for j in range(cand_len):
                    M[THIS_ROW * offset + j] = 0
                    WS[THIS_ROW_WS * offset + j] = 0

                # Delete from query to candidate
                M[THIS_ROW * offset + cand_len] = (i + 1) * self.delete_penalty
                WS[THIS_ROW_WS * offset + cand_len] = (i + 1) * self.delete_penalty

                # print("Two_AGO")
                # for j in range(offset):
                #     print("{}".format(storage[TWO_AGO * offset + j]), end=" ")
                #
                # print("\n ONE_AGO")
                # for j in range(offset):
                #     print("{}".format(storage[ONE_AGO * offset + j]), end=" ")
                #
                # print("\n THIS_ROW")
                # for j in range(offset):
                #     print("{}".format(storage[THIS_ROW * offset + j]), end=" ")
                #
                # print("")

                # now compute costs
                for j in range(cand_len):
                    delete_cost = M[ONE_AGO * offset + j] + self.delete_penalty

                    add_cost = M[THIS_ROW * offset + (j - 1 if j > 0 else cand_len)] + self.insert_penalty

                    subtract_cost = M[ONE_AGO * offset + (j - 1 if j > 0 else cand_len)] + (
                            query[i] != candidate[j]) * self.subs_penalty
                    # print("\t{} + {}".format(storage[ONE_AGO * offset + j], delete_penalty))
                    # print("\t{} + {}".format(storage[THIS_ROW * offset + (j - 1 if j > 0 else cand_len)], insert_penalty))
                    # print("\t{} + {}".format(storage[ONE_AGO * offset + (j - 1 if j > 0 else cand_len)], (query[i] != candidate[j]) * subs_penalty))

                    M[THIS_ROW * offset + j] = min(delete_cost, add_cost, subtract_cost)
                    WS[THIS_ROW_WS * offset + j] = min(WS[ONE_AGO_WS * offset + j] + self.delete_penalty,
                                                       WS[THIS_ROW_WS * offset + (
                                                           j - 1 if j > 0 else cand_len)] + self.insert_penalty,
                                                       WS[ONE_AGO_WS * offset + (
                                                           j - 1 if j > 0 else cand_len)] + self.subs_penalty)

                    # print("i={} j={} ({},{},{}) {}".format(i, j, WS[ONE_AGO_WS * offset + j] + delete_penalty, WS[THIS_ROW_WS * offset + (
                    #                                         j - 1 if j > 0 else cand_len)] + insert_penalty, WS[ONE_AGO_WS * offset + (
                    #                                         j - 1 if j > 0 else cand_len)] + subs_penalty,
                    #                                        WS[THIS_ROW_WS * offset + j]))

                    # deal with transpositions
                    if self.enable_trans and i > 0 and j > 0 and query[i] == candidate[j - 1] and query[i - 1] == \
                            candidate[j] and query[i] != candidate[j]:
                        # print("\tTransposition: min({},{})".format(storage[THIS_ROW * offset + j], storage[
                        #     TWO_AGO * offset + j - 2 if j > 1 else cand_len] + trans_penalty))
                        M[THIS_ROW * offset + j] = min(M[THIS_ROW * offset + j],
                                                       M[
                                                           TWO_AGO * offset + j - 2 if j > 1 else cand_len] + self.trans_penalty)
                        # print("\tNew Value: {}".format(storage[THIS_ROW * offset + j]))

            # prevent division by zero for empty inputs
            # print("OP: {}".format(storage[THIS_ROW * offset + (cand_len - 1)]))
            # printf("SIM: %f\tmin=%f , max=%f\t Normalized sim: %f\n", M[THIS_ROW * offset + (cand_len - 1)], 0,
            #                                                              WS[THIS_ROW_WS * offset + (cand_len - 1)],
            #                                                              M[THIS_ROW * offset + (cand_len - 1)] / WS[THIS_ROW_WS * offset + (cand_len - 1)])
            return 1.0 - (M[THIS_ROW * offset + (cand_len - 1)] / WS[THIS_ROW_WS * offset + (cand_len - 1)])
        finally:
            # free dynamically-allocated memory
            free(M)
            free(WS)
