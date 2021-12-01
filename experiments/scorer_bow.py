import logging

from data.preprocessing import preprocess_stacktrace, retrieve_df, std_function_preprocess
from cython_mod.cmp_func_bow import compare_bow
from nltk import ngrams


class ScorerBoW(object):

    def __init__(self, bug_db, method, arguments, max_depth, nthreads, filter_recursion, keep_ukn, static_df_ukn,
                 rm_dup_stacks, freq_by_stacks, n_gram=1, preprocess_fn=std_function_preprocess):
        self.bug_db = bug_db
        self.method = method

        self.arguments = arguments
        self.stacktraces_by_id = {}
        self.nthreads = nthreads
        self.max_depth = max_depth if max_depth > 0 else 9999999
        self.filter_recursion = filter_recursion
        self.reports = None
        self.unique_ukn_report = not keep_ukn
        self.ukn_set = set() if static_df_ukn else None
        self.preprocess_fn = preprocess_fn
        self.rm_dup_stacks = rm_dup_stacks
        self.n_gram = n_gram
        self.freq_by_stacks = freq_by_stacks

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

        # Merge stacks and transform to BOW
        for bug_id, stacks in list(self.stacktraces_by_id.items()):
            new_stack = []

            for st in stacks:
                new_stack.extend(st)
                for ngram in range(2, self.n_gram+1):
                    new_stack.extend((vocab.setdefault(t, len(vocab)) for t in ngrams(st,ngram)))

            self.stacktraces_by_id[bug_id] = [sorted(new_stack)]

        for stack_id, stacks in self.stacktraces_by_id.items():
            if len(stacks) != 1:
                raise Exception(f"Report should have one stack: {stack_id}")

        self.reports = report_ids - set(queries)
        self.function_df, self.n_docs = retrieve_df((self.stacktraces_by_id[id] for id in self.reports), extra=len(vocab),
                                       freq_by_stacks=self.freq_by_stacks)

    def add_report(self, query_id):
        if query_id not in self.reports:
            query_stacktrace = self.stacktraces_by_id[query_id]
            self.function_df, self.n_docs = retrieve_df([query_stacktrace], True, self.function_df, 5000, self.freq_by_stacks, self.n_docs)
            self.reports.add(query_id)

    def score(self, query_id, candidate_ids):
        # Get the first stack trace from query
        query_stacktrace = self.stacktraces_by_id[query_id]
        candidate_stacktraces = [self.stacktraces_by_id[cand_id] for cand_id in candidate_ids]

        if len(query_stacktrace[0]) == 0:
            raise Exception(f"Empty query: {query_id}")

        if query_id not in self.reports:
            self.function_df, self.n_docs = retrieve_df([query_stacktrace], True, self.function_df, 5000, self.freq_by_stacks, self.n_docs)
            self.reports.add(query_id)

        idf = (self.n_docs/(self.function_df + 0.00000000001))

        if self.ukn_set:
            for tkn_idx in self.ukn_set:
                if tkn_idx < len(idf):
                    idf[tkn_idx] = 0.0000001

        scores = compare_bow(self.nthreads, self.method, self.arguments, query_stacktrace, candidate_stacktraces, idf)

        return scores
