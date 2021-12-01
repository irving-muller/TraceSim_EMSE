import argparse
import json
import logging
import math
from time import time

import numpy as np
from collections import defaultdict
from itertools import chain
from multiprocessing.pool import Pool

from gensim.matutils import corpus2csc
from scipy.sparse import vstack

from data.preprocessing import preprocess_stacktrace, retrieve_df, generate_doc_freq_matrix
from data.report_dataset import ReportDataset
from data.bug_report_database import BugReportDatabase
from evaluation.eval_strategy import SunStrategy, generate_recommendation_list, RecommendationFixer
from evaluation.metric import MAP_RecallRate, BinaryPredictionROC
from util.jsontools import JsonLogFormatter

from experiments.scorer import Scorer
from cython_mod.method.method import SimilarityMethod


def run_opt_align(args, report_db, validationDataset):
    return run(args, report_db, validationDataset, 'opt_align')

def run_damerau(args, report_db, validationDataset):
    return run(args, report_db, validationDataset, 'damerau')

def run(args, report_db, validationDataset, alg):
    # logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
    eval_strategy = SunStrategy(report_db, validationDataset, args.w)
    if alg == 'opt_align':
        arguments = [args.insert_penalty, args.subs_penalty, args.match_cost]
        scorer = Scorer(report_db, SimilarityMethod.OPT_ALIGN, args.aggregate, arguments, args.max_depth,
                        args.nthreads, args.filter_recursion, args.df_threshold, args.field_coef, args.keep_ukn,
                        args.static_df_ukn, args.rm_dup_stacks, args.filter, args.filter_k, args.filter_func,
                        args.filter_func_k, args.freq_by_stacks)
    elif alg == 'damerau':
        arguments = [args.insert_penalty, args.delete_penalty, args.subs_penalty, args.trans_cost, args.disable_trans]
        scorer = Scorer(report_db, SimilarityMethod.DAMERAU_LEVENSHTEIN, args.aggregate, arguments, args.max_depth,
                        args.nthreads, args.filter_recursion, args.df_threshold, args.field_coef, args.keep_ukn,
                        args.static_df_ukn, args.rm_dup_stacks, args.filter, args.filter_k, args.filter_func,
                    args.filter_func_k, args.freq_by_stacks)
    recommendation_file = generate_recommendation_list(args.result_file, eval_strategy, scorer, args.sparse)
    listeners = [
        MAP_RecallRate(report_db, args.k, True),
        BinaryPredictionROC(report_db, eval_strategy.get_report_set()),
    ]
    recommendation_fixer = RecommendationFixer(eval_strategy, (args.sparse or (args.result_file is None or len(args.result_file) == 0)))
    recommendation_file.iterate(eval_strategy, listeners, recommendation_fixer)

    return [l.compute() for l in listeners]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg', help="Options: opt_align () or damerau")
    parser.add_argument('bug_dataset', help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('evaluation', help="Evaluation dataset with pairs")

    parser.add_argument('-insert_penalty', default=1, type=float, help="")
    parser.add_argument('-delete_penalty', default=1, type=float, help="")
    parser.add_argument('-subs_penalty', default=1, type=float, help="")
    parser.add_argument('-match_cost', default=1, type=float, help="")
    parser.add_argument('-trans_cost', default=1, type=float, help="")

    parser.add_argument('-field_coef', default=0.0, type=float, help="")

    parser.add_argument('-t', nargs='+', type=float, help="threshold dataset with pairs")
    parser.add_argument('-k', nargs='+', type=int, help="Recall rate @k")
    parser.add_argument('-w', help="Time window")
    parser.add_argument('-disable_trans', action="store_false")

    parser.add_argument('-df_threshold', default=0.0, type=float, help="threshold dataset with pairs")
    parser.add_argument('-result_file', help="Time window")

    parser.add_argument('-max_depth', type=int, default=300, help="Validation dataset with pairs")
    parser.add_argument('-nthreads', type=int, default=5, help="Validation dataset with pairs")
    parser.add_argument('-filter_recursion', help="Options: none, modani and brodie")
    parser.add_argument('-keep_ukn', action="store_true", help="")
    parser.add_argument('-static_df_ukn', action="store_true", help="")
    parser.add_argument('-aggregate', default="max",
                        help="Options: max, avg_query, avg_cand, avg_short, avg_long w_avg_query, w_avg_short, w_max_query")
    parser.add_argument('-rm_dup_stacks', action="store_true", help="")
    parser.add_argument('-sparse', action="store_true", help="")
    parser.add_argument('-filter', default=None, help="Options: none, select_one, top_k")
    parser.add_argument('-filter_k', default=-1.0, type=float, help="Percentage of the k-top functions")
    parser.add_argument('-filter_func', default=None, help="Options: none, top_k, top_k_trim")
    parser.add_argument('-filter_func_k', default=-1.0, type=float, help="Percentage of the k-top functions")
    parser.add_argument('-freq_by_stacks', action="store_true", help="")

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logHandler = logging.StreamHandler()
    formatter = JsonLogFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    logger.info(args)

    report_db = BugReportDatabase.from_json(args.bug_dataset)
    validationDataset = ReportDataset(args.evaluation)

    results = run(args, report_db, validationDataset, args.alg)

    for r in results:
        logger.info(r)

    print(json.dumps(results))