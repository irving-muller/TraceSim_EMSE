# distutils: language = c++
# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp


from libcpp.list cimport  list
from libcpp.vector cimport vector
from libc.stdlib cimport calloc, free
from cython.operator cimport postincrement, postdecrement, dereference
from cpython cimport array
import array

cimport cython_mod.util.structures
from cython_mod.util.structures cimport Stacktrace
from cython.parallel import prange

from libc.stdio cimport printf


cimport numpy as np
import numpy as np


cdef to_c_by_report(vector[vector[Stacktrace]] & stacks_by_reportid, report_stacks, np.uint8_t[:] is_stop_word, bint beg_trail_trim ):
    cdef Py_ssize_t report_idx
    cdef vector[Stacktrace] stacks

    for report_idx in range(len(report_stacks)):
        stacks = vector[Stacktrace]()

        stacks.reserve(len(report_stacks[report_idx]))

        to_c(stacks, report_idx, report_stacks[report_idx], is_stop_word, beg_trail_trim)
        stacks_by_reportid.push_back(stacks)


cdef void strip(list[int] & st, np.uint8_t[:] & is_stop_word, bint beg_trail_trim):
    cdef int function_id
    cdef list[int].iterator it
    cdef list[int].iterator rit

    it = st.begin()
    # Trim forward
    while it != st.end():
        function_id = dereference(it)
        # printf("\t(%d,%d)\n", function_id, is_stop_word[function_id])
        if is_stop_word[function_id]:
            # printf("\t\tErased\n")
            it = st.erase(it)
        else:
            # We found a non stop word
            if beg_trail_trim:
                break
            else:
                postincrement(it)

    if it != st.end():
        # Trim backward
        rit = st.end()
        postdecrement(rit)
        # print("Backward")

        while rit != it:
            function_id = dereference(rit)
            # printf("\t(%d,%d)\n", function_id, is_stop_word[function_id])

            if is_stop_word[function_id]:
                # printf("\t\tErased\n")
                rit = st.erase(rit)
                postdecrement(rit)
            else:
                break




cdef to_c(vector[Stacktrace] & stacks, int report_idx, report_stack, np.uint8_t[:] is_stop_word, bint beg_trail_trim):
    cdef int n_reports;
    cdef Py_ssize_t stack_list_idx, stack_idx
    cdef int * stack_c
    cdef Stacktrace s
    cdef list[int] st
    cdef int function_id


    for stack_list_idx in range(len(report_stack)):
        st = report_stack[stack_list_idx]

        # print("Before filter")
        # a = ""
        # for k in range(len(report_stack[stack_list_idx])):
        #     a+= "{}, ".format(report_stack[stack_list_idx][k])
        # print("\t{}".format(a))


        if is_stop_word is not None or is_stop_word.shape[0] != 0:
            # print("Filter functions")
            strip(st, is_stop_word, beg_trail_trim)

        if st.size() == 0:
            # print("Filtered all functions")
            continue

        stack_c =  <int *> calloc(st.size(), sizeof(int))

        stack_idx = 0
        for function_id in st:
            stack_c[stack_idx] = function_id
            stack_idx+=1

        s.stack = stack_c
        s.report_idx = report_idx
        s.length = st.size()

        # printf("After filter\n\t")
        # for stack_idx in range(s.length):
        #     printf("%d ",s.stack[stack_idx])
        # printf("\n")

        stacks.push_back(s)



cdef clean(vector[Stacktrace] & query_stacks, vector[vector[Stacktrace]] & cand_stacks):
    cdef Py_ssize_t i, j
    cdef Stacktrace * st
    cdef double cur_score, score
    cdef Py_ssize_t offset
    cdef vector[Stacktrace] * v

    for i in range(query_stacks.size()):
        free(query_stacks[i].stack)

    for i in range(cand_stacks.size()):
        v = &cand_stacks[i]

        for j in range(v.size()):
            st = &v.at(j)
            free(st.stack)




