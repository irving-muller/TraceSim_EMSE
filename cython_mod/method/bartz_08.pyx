from cpython.version cimport PY_MAJOR_VERSION
from libc.stdlib cimport calloc, free

import numpy
cimport numpy

ctypedef numpy.uint32_t DTYPE_t

cdef Py_ssize_t ONE_AGO = 0
cdef Py_ssize_t THIS_ROW = 1

cdef bint same_module(int mod1, int mod2, int ukn_id, bint ukn_same_group):
    if mod1 == ukn_id and mod2 == ukn_id:
        return ukn_same_group

    return mod1 == mod2

cdef bint same_function(int func1, int func2, int ukn_id):
    if func1 == ukn_id and func2 == ukn_id:
        return False

    print("\tfunc {} == {}".format(func1, func2))
    return func1 == func2

cpdef float bartz_08(numpy.ndarray[DTYPE_t, ndim=2] seq1, numpy.ndarray[DTYPE_t, ndim=2] seq2, float ins_same,
                     float ins_new, float del_same, float del_last, float sub_mod, float sub_func, int ukn_id,
                     bint ukn_same_group):
    cdef float n

    cdef Py_ssize_t s2_len = seq2.shape[0]
    cdef Py_ssize_t s1_len = seq1.shape[0]

    # Py_ssize_t should be used wherever we're dealing with an array index or length
    cdef Py_ssize_t i, j
    cdef Py_ssize_t offset = s2_len + 1
    cdef float delete_cost, insert_cost, subtract_cost, edit_distance

    # storage is a 3 x (len(seq2) + 1) array that stores TWO_AGO, ONE_AGO, and THIS_ROW
    cdef float *M = <float *> calloc(2 * offset, sizeof(float))
    cdef double *WS = <double *> calloc(2 * offset, sizeof(double))
    if not M:
        raise MemoryError()

    # Largest penalty values for each operation
    cdef float default_ins_penalty = ins_new if ins_new > ins_same else ins_same
    cdef float default_del_penalty = del_last if del_last > del_same else del_same
    cdef float default_subs_penalty = sub_mod if sub_mod > sub_func else sub_func

    cdef float delete_penalty, insert_penalty, subs_penalty

    try:
        # initialize THIS_ROW
        for i in range(1, offset):
            # Insert from query to candidate
            M[THIS_ROW * offset + (i - 1)] = i * default_ins_penalty
            WS[THIS_ROW * offset + (i - 1)] = i * default_ins_penalty

        # print("I: {} {}".format(s1_len, s2_len))
        for i in range(s1_len):
            # swap/initialize vectors
            for j in range(offset):
                M[ONE_AGO * offset + j] = M[THIS_ROW * offset + j]
                WS[ONE_AGO * offset + j] = WS[THIS_ROW * offset + j]

            for j in range(s2_len):
                M[THIS_ROW * offset + j] = 0
                WS[THIS_ROW * offset + j] = 0

            # Delete from query to candidate
            M[THIS_ROW * offset + s2_len] = (i + 1) * default_del_penalty
            WS[THIS_ROW * offset + s2_len] = (i + 1) * default_del_penalty

            # now compute costs
            for j in range(s2_len):
                is_same_module = same_module(seq1[i][1], seq2[j][1], ukn_id, ukn_same_group)

                if is_same_module or (
                        j + 1 < s2_len and same_module(seq1[i][1], seq2[j + 1][1], ukn_id, ukn_same_group)):
                    delete_penalty = del_same
                    # print("Del same")
                else:
                    delete_penalty = del_last
                    # print("Del last")

                delete_cost = M[ONE_AGO * offset + j] + delete_penalty

                is_same_function = same_function(seq1[i][0], seq2[j][0], ukn_id)
                # print("\ti={} j={} {} {}".format(seq1[i], seq2[j], is_same_module, is_same_function))

                if is_same_module or (
                        i + 1 < s1_len and same_module(seq1[i + 1][1], seq2[j][1], ukn_id, ukn_same_group)):
                    insert_penalty = ins_same
                    # print("Ins same")
                else:
                    insert_penalty = ins_new
                    # print("Ins new")

                insert_cost = M[THIS_ROW * offset + (j - 1 if j > 0 else s2_len)] + insert_penalty

                if not is_same_module:
                    subs_penalty = sub_mod
                    # print("Sub mod")
                elif not is_same_function:
                    subs_penalty = sub_func
                    # print("Sub func")
                else:
                    subs_penalty = 0

                subtract_cost = M[ONE_AGO * offset + (j - 1 if j > 0 else s2_len)] + subs_penalty

                M[THIS_ROW * offset + j] = min(insert_cost,  subtract_cost,delete_cost)
                WS[THIS_ROW * offset + j] = min(
                    WS[THIS_ROW * offset + (j - 1 if j > 0 else s2_len)] + default_ins_penalty,
                    WS[ONE_AGO * offset + (j - 1 if j > 0 else s2_len)] + default_subs_penalty,
                    WS[ONE_AGO * offset + j] + default_del_penalty)
                # print("i={} j={} ({},{},{}) {}".format(i, j, insert_cost, subtract_cost, delete_cost,
                #                                        M[THIS_ROW * offset + j]))

        # prevent division by zero for empty inputs
        # print("SIM: {}\tmin={} , max={}\t Normalized sim: {}".format(M[THIS_ROW * offset + (s2_len - 1)], 0,
        #                                                              WS[THIS_ROW * offset + (s2_len - 1)],
        #                                                              float(M[THIS_ROW * offset + (s2_len - 1)]) / WS[
        #                                                                  THIS_ROW * offset + (s2_len - 1)]))
        return float(M[THIS_ROW * offset + (s2_len - 1)]) / WS[THIS_ROW * offset + (s2_len - 1)]
    finally:
        # free dynamically-allocated memory
        free(M)
        free(WS)
