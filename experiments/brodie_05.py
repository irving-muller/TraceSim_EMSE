import argparse
import json
import logging

from cython_mod.method.method import SimilarityMethod
from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset
from evaluation.eval_strategy import SunStrategy, generate_recommendation_list, RecommendationFixer
from evaluation.metric import MAP_RecallRate, BinaryPredictionROC
from experiments.scorer import Scorer
from util.jsontools import JsonLogFormatter


def run(args, report_db,validationDataset):
    global results
    # logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
    eval_strategy = SunStrategy(report_db, validationDataset, args.w)
    arguments = [args.coef_gap, args.gap_penalty, args.mismatch_penalty]
    scorer = Scorer(report_db, SimilarityMethod.BRODIE_05, args.aggregate, arguments, args.max_depth,
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
    parser.add_argument('bug_dataset', help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('evaluation', help="Evaluation dataset with pairs")
    parser.add_argument('-coef_gap', required=True, type=float, help="threshold dataset with pairs")
    parser.add_argument('-gap_penalty', required=True, type=float, help="threshold dataset with pairs")
    parser.add_argument('-mismatch_penalty', default=0, type=float, help="threshold dataset with pairs")
    parser.add_argument('-df_threshold', default=0.0, type=float, help="threshold dataset with pairs")
    parser.add_argument('-field_coef', default=0.0, type=float, help="")
    parser.add_argument('-t', nargs='+', type=float, help="threshold dataset with pairs")
    parser.add_argument('-k', nargs='+', type=int, help="Recall rate @k")
    parser.add_argument('-w', help="Time window")
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

    results = run(args, report_db,validationDataset)

    for r in results:
        logger.info(r)

    print(json.dumps(results))
