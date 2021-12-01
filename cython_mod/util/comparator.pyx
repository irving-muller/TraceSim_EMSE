# distutils: language = c++
# cython: language_level=3

from cython_mod.util.structures cimport Stacktrace
from libc.stdlib cimport calloc, free, malloc
from libcpp.algorithm cimport sort
from libc.stdio cimport printf
from libcpp.vector cimport vector
import numpy
cimport numpy

cimport cython

from cython_mod.method.method cimport DTYPE_t, Method

cdef class Comparator:

    cdef vector[StackTraceInfo] prepare(self, vector[Stacktrace] & stacks, vector[Stacktrace] & other_stacks) nogil:
        cdef Py_ssize_t i
        cdef vector[StackTraceInfo] v
        cdef StackTraceInfo info

        v.reserve(stacks.size())

        for i in range(stacks.size()):
            info = StackTraceInfo()

            info.idx = i
            info.weight = 1.0
            info.stack = &stacks.at(i)

            v.push_back(info)

        return v

    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks) nogil:
        pass


cdef class Max(Comparator):

    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks) nogil:
        cdef double result=0.0
        cdef double max_result = -99999999.0
        cdef Py_ssize_t i, j

        for i in range(query_stacks.size()):
            for j in range(candidate_stacks.size()):
                result = matrix_score[(i * candidate_stacks.size()) + j]
                if max_result < result:
                    max_result = result
                # printf("\t%f", result)
            # printf("\n")

        # printf("\tMAX: %f\n", max_result)
        return max_result


cdef class Mean(Comparator):
    def __cinit__(self, MeanType mean_type, WeightType weight_type, DTYPE_t[:] & df_array):
        self.mean_type = mean_type
        self.weight_type = weight_type
        self.df_array = df_array

    cdef vector[StackTraceInfo] prepare(self, vector[Stacktrace] & stacks, vector[Stacktrace] & other_stacks) nogil:
        cdef Py_ssize_t i
        cdef vector[StackTraceInfo] v
        cdef StackTraceInfo info

        v.reserve(stacks.size())

        for i in range(stacks.size()):
            info = StackTraceInfo()

            info.idx = i
            info.stack = &stacks.at(i)

            if self.weight_type == WeightType.AVG:
                info.weight = compute_weight(info.stack,self.df_array)
            elif self.weight_type == WeightType.MAXI:
                info.weight = compute_weight_max(info.stack,self.df_array)
            elif self.weight_type == WeightType.OFF:
                info.weight = 1.0

            v.push_back(info)

        return v

    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks) nogil:
        # print("N candidates: {}".format(self.n_candidates))
        cdef int avg_row
        cdef int length

        if self.mean_type == MeanType.QUERY:
            avg_row = 1
            length = query_stacks.size()
        elif self.mean_type == MeanType.CANDIDATE:
            avg_row = 0
            length = candidate_stacks.size()
        elif self.mean_type == MeanType.SHORTEST:
            length= min(candidate_stacks.size(), query_stacks.size())
            avg_row = query_stacks.size() <= candidate_stacks.size()
        elif self.mean_type == MeanType.LONGEST:
            length= max(candidate_stacks.size(), query_stacks.size())
            avg_row = candidate_stacks.size() <= query_stacks.size()
        elif self.mean_type == MeanType.QUERY_CAND:
            length=  query_stacks.size() + candidate_stacks.size()
            avg_row = -1


        cdef double * max_values = <double *> calloc(length, sizeof(double))
        cdef double score
        cdef Py_ssize_t query_idx, cand_idx

        cdef Py_ssize_t k
        for k in range(length):
            max_values[k] = -99999999.0

        for query_idx in range(query_stacks.size()):
            for cand_idx in range(candidate_stacks.size()):
                score = matrix_score[(query_idx * candidate_stacks.size()) + cand_idx]

                if avg_row == -1:
                    if score > max_values[query_idx]:
                        max_values[query_idx] = score

                    if score > max_values[cand_idx + query_stacks.size()]:
                        max_values[cand_idx  + query_stacks.size()] = score
                elif avg_row == 1:
                    if score > max_values[query_idx]:
                        max_values[query_idx] = score
                elif  avg_row == 0:
                    if score > max_values[cand_idx]:
                        max_values[cand_idx] = score

        cdef double sum = 0.0
        cdef double sum2 = 0.0
        cdef double den = 0.0
        cdef double den2 = 0.0
        cdef double weight
        cdef Py_ssize_t i

        for i in range(length):
            if avg_row == -1:
                if i < query_stacks.size():
                    weight = query_stacks[i].weight
                    sum += weight * max_values[i]
                    den += 2.0 * weight
                else:
                    weight = candidate_stacks[i - query_stacks.size()].weight
                    sum2 += weight * max_values[i]
                    den2 += 2.0 * weight
            else:
                if avg_row == 1:
                    weight = query_stacks[i].weight
                elif avg_row == 0:
                    weight = candidate_stacks[i].weight

                sum += weight * max_values[i]
                den += weight

                den2 = 1.0

        free(max_values)
        # printf("\n###### sum: %f sum2:%f den: %f, den2: %f\n", sum, sum2, den, den2)

        return sum/den + sum2/den2


cdef bint compare_align_elem(DfInfo o1, DfInfo o2) nogil:
    if o1.max_value == o2.max_value:
        return o1.mean_value < o2.mean_value
    else:
       return o1.max_value < o2.max_value


cdef bint compare_align_double(double o1, double o2) nogil:
    return o1 < o2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double compute_weight(Stacktrace * stack, DTYPE_t[:] & df_array) nogil:
    cdef Py_ssize_t i
    cdef double sum = 0.0

    for i in range(stack.length):
       sum+= 100.0 - df_array[stack.stack[i]]

    return sum/stack.length

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double  compute_weight_max(Stacktrace * stack, DTYPE_t[:] & df_array) nogil:
    cdef Py_ssize_t i
    cdef Stacktrace * st
    cdef double max_val, s
    cdef int func_id

    max_val = -1.0

    # printf("IDF:")
    for i in range(stack.length):
        s = 100.0 - df_array[stack.stack[i]]
        # printf(" (%d, %f)", stack.stack[i], s)

        if s > max_val:
            max_val = s
    # printf("\n")



    return max_val


cdef class Filter:
    cdef vector[Stacktrace] filter(self, vector[Stacktrace] & stacks) nogil:
        return stacks

cdef class SelectOne(Filter):
    # Select the most significant function

    def __cinit__(self, DTYPE_t[:] & df_array):
        self.df_array = df_array

    cdef vector[Stacktrace] filter(self, vector[Stacktrace] & stacks) nogil:
        cdef Py_ssize_t i

        if stacks.size() == 1:
            return stacks

        cdef vector[Stacktrace] new_stacks
        cdef double max_idf = -1.0
        cdef double max_mean_idf = -1.0
        cdef Py_ssize_t best_idx = -1

        cdef double max_value
        cdef double mean_value

        for i in range(stacks.size()):
            max_value = compute_weight_max(&stacks[i], self.df_array)
            mean_value = compute_weight(&stacks[i], self.df_array)

            # printf("W: %d mean=%f max=%f\n", i, mean_value, max_value)
            if max_value > max_idf or (max_value == max_idf and mean_value > max_mean_idf):
                max_idf = max_value
                max_mean_idf = mean_value
                best_idx = i

        # printf("MAX: %d mean=%f max=%f\n", best_idx, max_mean_idf, max_idf)
        new_stacks.push_back(stacks[best_idx])

        return new_stacks

cdef class KTopFunction(Filter):

    def __cinit__(self, double filter_k, DTYPE_t[:] & df_array):
        self.df_array = df_array

        cdef double * idfs  = <double * > malloc(df_array.shape[0] *  sizeof(double))
        cdef Py_ssize_t i
        cdef int vocab_size = df_array.shape[0]

        for i in range(vocab_size):
            idfs[i] = 100 - df_array[i]

        sort(idfs, idfs + vocab_size)

        if filter_k <= 0.0 or filter_k > 1.0:
            raise Exception("filter_k {} is invalid".format(filter_k))

        cdef int start = (int)((1.0-filter_k) * vocab_size)

        self.threshold = idfs[start]
        free(idfs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef vector[Stacktrace] filter(self, vector[Stacktrace] & stacks) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef double s

        cdef vector[Stacktrace] new_stacks
        cdef Stacktrace * st

        for i in range(stacks.size()):

            st = &stacks[i]

            for j in range(st.length):
                s = 100.0 - self.df_array[st.stack[j]]

                if s >= self.threshold:
                    new_stacks.push_back(stacks[i])
                    break

        return new_stacks

