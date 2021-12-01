import logging

from data.preprocessing import preprocess_stacktrace, retrieve_df, std_function_preprocess
from cython_mod.cmp_func import compare

from cython_mod.util.comparator import AggStrategy, FilterStrategy


class Scorer(object):

    def __init__(self, bug_db, method, agg_strategy, arguments, max_depth, nthreads, filter_recursion, df_threshold,
                 field_coef, keep_ukn, static_df_ukn, rm_dup_stacks, filter_strategy, filter_k, filter_func_strategy,
                 filter_function_k, freq_by_stacks, preprocess_fn=std_function_preprocess):
        self.bug_db = bug_db
        self.method = method
        self.filter_k = filter_k
        self.freq_by_stacks = freq_by_stacks
        self.filter_function_k = filter_function_k

        logger = logging.getLogger()

        if filter_strategy is None or filter_strategy == 'none':
            logger.info("Filter stack traces: None")
            self.filter_strategy = FilterStrategy.NONE
        elif filter_strategy == 'select_one':
            logger.info("Filter stack traces: select_one")
            self.filter_strategy = FilterStrategy.SELECT_ONE
        elif filter_strategy == 'top_k':
            logger.info("Filter stack traces: top_k")
            self.filter_strategy = FilterStrategy.TOP_K_FUNC

        if filter_func_strategy is None or filter_func_strategy is "none":
            logger.info("Filter function: None")
            self.filter_function = False
            self.beg_trail_trim = False
        elif filter_func_strategy == "threshold_trim":
            logger.info("Filter function: threshold_trim")
            self.filter_function = True
            self.beg_trail_trim = True

        if self.filter_function:
            if self.filter_function_k <= 0.0:
                raise Exception(f"self.filter_function_k has to be positive.")

        if agg_strategy == 'max':
            logger.info("Agg stg: max")
            self.agg_strategy = AggStrategy.MAX
        elif agg_strategy == 'avg_query':
            logger.info("Agg stg: avg_query")
            self.agg_strategy = AggStrategy.AVG_QUERY
        elif agg_strategy == 'avg_cand':
            logger.info("Agg stg: avg_cand")
            self.agg_strategy = AggStrategy.AVG_CAND
        elif agg_strategy == 'avg_short':
            logger.info("Agg stg: avg_short")
            self.agg_strategy = AggStrategy.AVG_SHORT
        elif agg_strategy == 'avg_long':
            logger.info("Agg stg: avg_long")
            self.agg_strategy = AggStrategy.AVG_LONG
        elif agg_strategy == 'avg_query_cand':
            logger.info("Agg stg: avg_query_cand")
            self.agg_strategy = AggStrategy.AVG_QUERY_CAND
        elif agg_strategy == 'align':
            logger.info("Agg stg: Align")
            self.agg_strategy = AggStrategy.ALIGN

        self.arguments = arguments
        self.stacktraces_by_id = {}
        self.nthreads = nthreads
        self.max_depth = max_depth if max_depth > 0 else 9999999
        self.filter_recursion = filter_recursion
        self.df_threshold = df_threshold
        self.field_coef = field_coef
        self.reports = None
        self.unique_ukn_report = not keep_ukn
        self.ukn_set = set() if static_df_ukn else None
        self.preprocess_fn = preprocess_fn
        self.rm_dup_stacks = rm_dup_stacks

        if filter_recursion is not None:
            logging.getLogger("Filtering recursions. {}".format(self.filter_recursion))

    def preprocess_reports(self, report_ids, queries):
        self.stacktraces_by_id, vocab = preprocess_stacktrace(report_ids, self.bug_db, self.max_depth,
                                                              filter_recursion=self.filter_recursion, return_vocab=True,
                                                              merge_main_nested=True,
                                                              unique_ukn_report=self.unique_ukn_report,
                                                              ukn_set=self.ukn_set,
                                                              preprocess_func=self.preprocess_fn,
                                                              rm_dup_stacks=self.rm_dup_stacks)

        self.reports = report_ids - set(queries)
        self.function_df, self.n_docs = retrieve_df((self.stacktraces_by_id[id] for id in self.reports), extra=len(vocab),
                                       freq_by_stacks=self.freq_by_stacks)

    def add_report(self, query_id):
        if query_id not in self.reports:
            query_stacktrace = self.stacktraces_by_id[query_id]
            self.function_df, self.n_docs = retrieve_df([query_stacktrace], True, self.function_df, 5000,
                                                        self.freq_by_stacks, self.n_docs)
            self.reports.add(query_id)

    def score(self, query_id, candidate_ids):
        # Get the first stack trace from query
        query_stacktrace = self.stacktraces_by_id[query_id]
        candidate_stacktraces = [self.stacktraces_by_id[cand_id] for cand_id in candidate_ids]

        if query_id not in self.reports:
            self.function_df, self.n_docs = retrieve_df([query_stacktrace], True, self.function_df, 5000, self.freq_by_stacks, self.n_docs)
            self.reports.add(query_id)

        normalized_df = (self.function_df / self.n_docs) * 100.0

        if self.ukn_set:
            for tkn_idx in self.ukn_set:
                if tkn_idx < len(normalized_df):
                    normalized_df[tkn_idx] = 99.999

        is_stop_words = None

        if self.filter_function:
            # todo: should we remove the ukn functions
            is_stop_words = normalized_df > self.filter_function_k

            if self.ukn_set:
                for tkn_idx in self.ukn_set:
                    is_stop_words[tkn_idx] = False

        scores = compare(self.nthreads, self.method, self.agg_strategy, self.arguments, query_stacktrace,
                         candidate_stacktraces, normalized_df, self.filter_strategy, self.filter_k, is_stop_words,
                         self.beg_trail_trim)

        return scores
