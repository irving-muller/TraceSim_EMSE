# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector

cimport cython_mod.util.structures
from cython_mod.util.structures cimport Stacktrace
from cython_mod.method.method cimport DTYPE_t, Method

cimport numpy as np
import numpy as np


cdef to_c(vector[Stacktrace] & stacks, int report_idx, report_stack, np.uint8_t[:] is_stop_word, bint beg_trail_trim)
cdef to_c_by_report(vector[vector[Stacktrace]] & stacks_by_reportid, report_stacks, np.uint8_t[:] is_stop_word, bint beg_trail_trim)
cdef clean(vector[Stacktrace] & query_stacks, vector[vector[Stacktrace]] & cand_stacks)

