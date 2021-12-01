# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
"""
Position Dependent Model
"""

from libc.math cimport exp
from libc.stdlib cimport calloc, free
from libc.math cimport exp, fabs

cdef Py_ssize_t ONE_AGO = 0
cdef Py_ssize_t THIS_ROW = 1

from libc.stdio cimport printf

cdef class PDM(Method):

    def __cinit__(self, double c, double o):
        self.c = c
        self.o = o

    cdef double similarity(self, int * query, int * candidate, long query_len, long cand_len) nogil:
        cdef Py_ssize_t i, j
        # We need to keep only two rows of the matrix M

        cdef Py_ssize_t offset = cand_len
        cdef double *M = <double *> calloc(2 * offset, sizeof(unsigned long))

        # printf("c: %f\to: %f", self.c, self.o)

        cdef double previous_row, previous_col, previous_row_col, cost, sim
        cdef int q_call, c_call

        try:
            for i in range(query_len):
                q_call = query[i]

                for j in range(offset):
                    M[ONE_AGO * offset + j] = M[THIS_ROW * offset + j]

                for j in range(cand_len):
                    c_call = candidate[j]

                    previous_row = 0.0 if i == 0 else M[ONE_AGO * offset + j]
                    previous_col = 0.0 if j == 0 else M[THIS_ROW * offset + j - 1]
                    previous_row_col = 0.0 if j == 0 or i == 0 else M[ONE_AGO * offset + j - 1]

                    if q_call == c_call:
                        # Add cost
                        previous_row_col += exp(-self.c * min(i, j)) * exp(-self.o * fabs(i - j))
                        # print(previous_row_col)
                        # print("{} * {} = {}".format(exp(-c * min(i, j)), exp(-o * abs(i - j)),
                        #                             exp(-c * min(i, j)) * exp(-o * abs(i - j))))

                    M[THIS_ROW * offset + j] = max(previous_row, previous_col, previous_row_col)
                    # print("i={} j={} ({},{},{}) {}".format(i,j, previous_row, previous_col, previous_row_col, M[THIS_ROW * offset + j]))
                # print("")
            sim = M[THIS_ROW * offset + cand_len - 1]
        finally:
            free(M)

        cdef double dividend = 0.0

        for j in range(min(query_len, cand_len)):
            dividend += exp(-self.c * j)

        return sim / dividend
