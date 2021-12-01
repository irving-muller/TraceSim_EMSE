# distutils: language = c++
# cython: language_level=3


cdef struct Stacktrace:
    int * stack
    int report_idx
    long length


cdef struct TermFreq:
    int term
    double freq

