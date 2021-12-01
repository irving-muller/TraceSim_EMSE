from cython_mod.util.structures cimport Stacktrace
from libcpp.vector cimport vector

from cython_mod.method.method cimport DTYPE_t, Method

cpdef enum AggStrategy:
    MAX=0
    AVG_QUERY=1
    AVG_CAND=2
    AVG_SHORT=3
    AVG_LONG=4
    AVG_QUERY_CAND=6


cdef struct ScoreWeight:
    double * scores
    double * weights
    bint source
    int len

cdef enum MeanType:
    QUERY, CANDIDATE, SHORTEST, LONGEST, SYM, QUERY_CAND

cdef enum WeightType:
    OFF, AVG, MAXI

cdef struct StackTraceInfo:
    int idx
    double weight
    Stacktrace * stack

cdef struct DfInfo:
    int idx
    double max_value
    double mean_value

cdef class Comparator:
    cdef vector[StackTraceInfo] prepare(self, vector[Stacktrace] & stacks, vector[Stacktrace] & other_stacks) nogil
    # cdef double compare(self, vector[Stacktrace] & query_stacks,  vector[Stacktrace] & cand_stacks, Method * method) nogil
    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks)  nogil

cdef class Max(Comparator):
    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks) nogil


cdef class Mean(Comparator):
    cdef MeanType mean_type
    cdef WeightType weight_type
    cdef DTYPE_t[:] df_array

    cdef vector[StackTraceInfo] prepare(self, vector[Stacktrace] & stacks, vector[Stacktrace] & other_stacks) nogil
    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks) nogil

cdef class Alignment(Comparator):
    cdef DTYPE_t[:] df_array

    cdef vector[StackTraceInfo] prepare(self, vector[Stacktrace] & stacks, vector[Stacktrace] & other_stacks) nogil
    cdef double aggregate(self, double * matrix_score, vector[StackTraceInfo] & query_stacks, vector[StackTraceInfo] & candidate_stacks) nogil


cdef bint compare_align_elem(DfInfo o1, DfInfo o2) nogil
cdef double compute_weight(Stacktrace * stack, DTYPE_t[:] & df_array) nogil
cdef double  compute_weight_max(Stacktrace * stack, DTYPE_t[:] & df_array) nogil


cpdef enum FilterStrategy:
    NONE=0
    SELECT_ONE=1
    TOP_K_FUNC=2


cdef class Filter:
    cdef vector[Stacktrace] filter(self, vector[Stacktrace] & stacks) nogil

cdef class SelectOne(Filter):
    cdef DTYPE_t[:] df_array

    cdef vector[Stacktrace] filter(self, vector[Stacktrace] & stacks) nogil

cdef class KTopFunction(Filter):
    cdef double threshold
    cdef  DTYPE_t[:] df_array

    cdef vector[Stacktrace] filter(self, vector[Stacktrace] & stacks) nogil
