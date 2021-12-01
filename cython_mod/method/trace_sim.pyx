# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from libc.stdlib cimport calloc, free
from libcpp.algorithm cimport sort
from libc.math cimport exp, fabs
from libc.stdio cimport printf


from cython.parallel import prange


import numpy
cimport numpy

cdef Py_ssize_t ONE_AGO = 0
cdef Py_ssize_t THIS_ROW = 1

cimport cython

from cython_mod.method.method cimport Method, DTYPE_t

cdef bint compare_frame(Frame o1, Frame o2) nogil:
    if o1.id == o2.id:
        return o1.weight < o2.weight
    else:
       return o1.id < o2.id

cdef void sum_equal_functions( Frame * frames, long seq_len) nogil:
    cdef int previous = -1
    cdef Frame f
    f.id = -1

    cdef Frame *first_frame
    cdef Frame *cur_frame
    cdef Frame *previous_frame = &f
    cdef Py_ssize_t idx;

    for idx in range(seq_len):
        cur_frame = frames + idx

        if previous_frame.id == cur_frame.id:
            if first_frame.id != cur_frame.id:
                printf("Ops.. something is wrong")

            first_frame.weight = first_frame.weight + cur_frame.weight
        else:
            first_frame = cur_frame

        previous_frame = cur_frame

cdef Frame * create_frames(int * seq, double * seq_weights, long seq_len) nogil:
    cdef Frame * frames = <Frame *> calloc(seq_len, sizeof(Frame))

    for pos in range(seq_len):
        frames[pos].id = seq[pos]
        frames[pos].weight = seq_weights[pos]

        # printf("\t%d %f\n", frames[pos].id, frames[pos].weight)

    return frames

cdef double calculate_den(int * query, double * q_pos_values, long query_len, int * candidate, double * c_pos_values, long cand_len) nogil:
    cdef Frame *q_frames = create_frames(query, q_pos_values, query_len)
    cdef Frame *c_frames = create_frames(candidate, c_pos_values, cand_len)

    sort(q_frames, q_frames + query_len, compare_frame)
    sort(c_frames, c_frames + cand_len, compare_frame)

    # Sum weight of same ids
    sum_equal_functions(q_frames, query_len)
    sum_equal_functions(c_frames, cand_len)

    cdef Py_ssize_t q_idx = 0
    cdef Py_ssize_t c_idx = 0
    cdef int current_id
    cdef double max_weight
    cdef int previous_id = -1
    cdef double den = 0.0
    cdef Frame * q_pointer
    cdef Frame * c_pointer

    while True:
        # printf("%d %d\n", q_idx, c_idx)
        q_pointer = q_frames + q_idx if q_idx < query_len else NULL
        c_pointer = c_frames + c_idx if c_idx < cand_len else NULL

        if q_pointer == NULL and c_pointer == NULL:
            break

        # printf("New It\n")
        if c_pointer == NULL or (q_pointer != NULL and q_pointer.id < c_pointer.id):
            # printf("Query\n")
            current_id = q_pointer.id
            max_weight = q_pointer.weight

            q_idx += 1
        elif q_pointer == NULL or (c_pointer != NULL and c_pointer.id < q_pointer.id):
            # printf("Cand\n")
            current_id = c_pointer.id
            max_weight = c_pointer.weight

            c_idx += 1
        elif c_pointer.id == q_pointer.id:
            # printf("ALL\n")
            current_id = c_pointer.id
            max_weight = max(q_pointer.weight, c_pointer.weight)

            q_idx += 1
            c_idx += 1


        if current_id != previous_id:
            # printf("Add weight: %f\n", max_weight)
            den+= max_weight
        # else:
        #     printf("No add weights\n")

        previous_id = current_id

    # printf("END\n")

    free(q_frames)
    free(c_frames)

    # printf("END2")

    return den



cdef class TraceSim(Method):

    def __cinit__(self, numpy.ndarray[DTYPE_t, ndim=1] df, double df_coef, double pos_coef, double diff_coef, double match_cost, double gap_penalty, double mismatch_penalty,
                  bint sigmoid, double gamma, bint sum_, bint idf, bint const_match, bint reciprocal_func, bint no_norm, bint const_gap, bint const_mismatch, bint brodie_function):
        self.df_coef = df_coef
        self.pos_coef = pos_coef
        self.diff_coef = diff_coef
        self.match_cost = match_cost
        self.gap_penalty = gap_penalty
        self.mismatch_penalty = mismatch_penalty
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.sum = sum_
        self.idf = idf
        self.const_match = const_match
        self.reciprocal_func = reciprocal_func
        self.no_norm = no_norm
        self.const_gap = const_gap
        self.const_mismatch = const_mismatch
        self.brodie_function = brodie_function

        if not self.sigmoid and self.idf:
            raise Exception("Exponential function cannot receive IDF")

        if self.brodie_function and self.idf:
            raise Exception("Brodie and IDF are not compatible")

        if self.idf:
            d = (df == 0.0) * 100.00 + df
            self.df = numpy.log(100/ d)
        elif self.brodie_function:
            self.df = df/100.0
        else:
            self.df = df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double calculate_weight(self, int * trace, Py_ssize_t pos, long seq_len) nogil:
        cdef double gw
        cdef double lw
        cdef double sigmoid_val

        if self.brodie_function:
            lw = 1.0 - pos / seq_len
            gw = 1.0 - self.df[trace[pos]]
        else:
            if self.reciprocal_func:
                lw = 1.0/((pos + 1)**self.pos_coef)
            else:
                lw = exp(-self.pos_coef * pos)

            if self.sigmoid:
                sigmoid_val= 1/(1 + exp(-self.df_coef * self.df[trace[pos]] + self.gamma))

                if not self.idf:
                    sigmoid_val = 1 - sigmoid_val

                gw = sigmoid_val
            else:
                gw = exp(-self.df_coef * self.df[trace[pos]])

        return lw * gw


    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil:
        cdef Py_ssize_t q_idx, c_idx
        # We need to keep only two rows of the matrix M

        cdef Py_ssize_t offset = cand_len + 1
        cdef double *M = <double *> calloc(2 * offset, sizeof(double))

        cdef double previous_row, previous_col, previous_row_col, cost, sim, normalized_sim
        cdef Py_ssize_t q_pos, c_pos
        cdef Py_ssize_t i, j, k

        cdef double *q_pos_values = <double *> calloc(query_len, sizeof(double))
        cdef double *c_pos_values = <double *> calloc(cand_len, sizeof(double))

        # printf("Query\n\t")
        # cdef double sum = 0.0
        for q_pos in range(query_len):
            q_pos_values[q_pos] = self.calculate_weight(query, q_pos, query_len)
        #     sum += q_pos_values[q_pos]
        #     printf("%f (%f, %f) ", q_pos_values[q_pos], exp(-self.df_coef * self.df[query[q_pos]]), exp(-self.pos_coef * q_pos))
        # printf("\n\tSum: %f", sum)

        # printf("\nCandidate\n\t")
        # sum = 0.0

        for c_pos in range(cand_len):
            c_pos_values[c_pos] = self.calculate_weight(candidate, c_pos, cand_len)
        #     sum += c_pos_values[c_pos]
        #     printf("%f  (%f, %f)", c_pos_values[c_pos], exp(-self.df_coef * self.df[candidate[c_pos]]), exp(-self.pos_coef * c_pos))
        # printf("\n\tSum: %f", sum)
        # printf("\n")

        # Worst scenario
        cdef double delete_cost, insert_cost, mismatch_cost
        # cdef double *WS = <double *> calloc(2 * offset, sizeof(double))

        # Variable to apply normalization
        cdef long shortest_len
        cdef double min_value, max_value
        cdef double den

        try:
            # Create first row
            for i in range(cand_len):
                M[THIS_ROW * offset + (i+1)] = M[THIS_ROW * offset + i] - self.gap_penalty * c_pos_values[i]
                # WS[THIS_ROW * offset + (i+1)] = WS[THIS_ROW * offset + i] - self.gap_penalty * c_pos_values[i]

            for i in range(query_len):
                q_pos = i
                i += 1

                # Copy THIS_ROW to ONE_AGO
                for k in range(offset):
                    M[ONE_AGO * offset + k] = M[THIS_ROW * offset + k]
                    # WS[ONE_AGO * offset + k] = WS[THIS_ROW * offset + k]

                # Reset THIS_ROW
                for k in range(offset):
                    M[THIS_ROW * offset + k] = 0.0
                    # WS[THIS_ROW * offset + k] = 0.0

                # Set first column of the row
                M[THIS_ROW * offset + 0] = M[ONE_AGO * offset + 0] - self.gap_penalty * q_pos_values[q_pos]
                # WS[THIS_ROW * offset + 0] = M[ONE_AGO * offset + 0] - self.gap_penalty * q_pos_values[q_pos]

                for j in range(cand_len):
                    c_pos = j
                    j += 1
                    # printf("%d   %d\n", i, j)

                    # Gap
                    if self.const_gap:
                        delete_cost = -self.gap_penalty
                        insert_cost = -self.gap_penalty
                    else:
                        delete_cost = -self.gap_penalty * q_pos_values[q_pos]
                        insert_cost = -self.gap_penalty * c_pos_values[c_pos]

                    previous_row = M[ONE_AGO * offset + j] + delete_cost
                    # printf("\tP_row (del) %f= %f - %f", previous_row, M[ONE_AGO * offset + j], delete_cost)
                    previous_col = M[THIS_ROW * offset + j - 1] + insert_cost
                    # printf("\tP_col (ins) %f= %f - %f",previous_col,  M[THIS_ROW * offset + j - 1], insert_cost)

                    previous_row_col = M[ONE_AGO * offset + j - 1]

                    if self.const_mismatch:
                        mismatch_cost = -self.mismatch_penalty
                    else:
                        if self.sum:
                            mismatch_cost = -self.mismatch_penalty * (q_pos_values[q_pos] + c_pos_values[c_pos])
                        else:
                            mismatch_cost = -self.mismatch_penalty * max(q_pos_values[q_pos], c_pos_values[c_pos])
                    # printf("\tmis %f; q_w=%f c_w=%f\n", mismatch_cost, q_pos_values[q_pos], c_pos_values[c_pos])

                    if query[q_pos] == candidate[c_pos]:
                        # IDF * Function call position * Shift between calls
                        if self.const_match:
                            cost = 1.0
                        else:
                            cost = max(q_pos_values[q_pos], c_pos_values[c_pos]) * exp(-self.diff_coef * fabs(<double>(q_pos - c_pos)))
                        # print("\tMACTH{}={}+ {} * {}: max({},{}) * {} ".format(
                        #     previous_row_col + match_cost * cost,
                        #     previous_row_col,
                        #     match_cost,
                        #     cost,
                        #     q_pos_values[q_pos], c_pos_values[c_pos], exp(-diff_coef * abs(q_pos - c_pos))))
                        previous_row_col += self.match_cost * cost
                    else:
                        # print("\tMIS_total{}={} - {} * {}: max({}, {}) ".format(
                        #     previous_row_col + mismatch_cost,
                        #     previous_row_col,
                        #     mismatch_penalty,
                        #     mismatch_cost/-mismatch_penalty,
                        #     q_pos_values[q_pos], c_pos_values[c_pos]))
                        previous_row_col += mismatch_cost

                    M[THIS_ROW * offset + j] = max(previous_row, previous_col, previous_row_col)

                    # WS[THIS_ROW * offset + j] = max(WS[ONE_AGO * offset + j] + delete_cost,
                    #                                 WS[THIS_ROW * offset + j - 1] + insert_cost,
                    #                                 WS[ONE_AGO * offset + j - 1] + mismatch_cost)

                    # print("\t\t WS i={} j={} ({},{},{}) {}".format(i, j, WS[ONE_AGO * offset + j], WS[ONE_AGO * offset + j - 1], WS[THIS_ROW * offset + j - 1],
                    #                                        WS[THIS_ROW * offset + j]))
                    # print("i={} j={} ({},{},{}) {}".format(i, j, previous_row, previous_row_col, previous_col,
                    #                                        M[THIS_ROW * offset + j]))
                # print("")

            sim = M[THIS_ROW * offset + offset - 1]

            # Min value
            # if cand_len > query_len:
            #     # shortest_seq = q_pos_values
            #     shortest_len = query_len
            # else:
            #     # shortest_seq = c_pos_values
            #     shortest_len = cand_len
            #
            # min_value = WS[THIS_ROW * offset + offset - 1]
            # max_value = 0
            #
            # for i in range(shortest_len):
            #     if self.const_match:
            #         max_value += 1.0
            #     else:
            #         max_value += max(q_pos_values[i], c_pos_values[i])
            #
            # max_value *= self.match_cost
            #
            # if max_value == min_value:
            #     return 0.0

            if not self.no_norm:
                den = calculate_den(query, q_pos_values, query_len, candidate, c_pos_values, cand_len)

                if den == 0.0:
                    normalized_sim = 0.0
                else:
                    normalized_sim = sim / den
            else:
                normalized_sim = sim


            # normalized_sim = (sim - min_value) / (max_value - min_value)

            # Jaccard similarity
            # printf("SIM: %f\tmin=%f , max=%f\t Normalized sim: %f\n", sim, min_value, max_value, (sim - min_value) / (max_value - min_value))
            # printf("SIM: %f\den=%f\t Normalized sim: %f\n", sim, den, normalized_sim)
            # print("SIM: {}\tmin={} , max={}\t Normalized sim: {}".format(sim, min_value, max_value, normalized_sim))

        finally:
            free(M)
            # free(WS)
            free(q_pos_values)
            free(c_pos_values)

        return normalized_sim
