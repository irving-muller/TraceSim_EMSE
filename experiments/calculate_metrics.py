import argparse
import logging

from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset
from evaluation.eval_strategy import SunStrategy, RecommendationFile, RecommendationFixer
from evaluation.metric import MAP_RecallRate, BinaryPredictionROC
from util.jsontools import JsonLogFormatter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bug_dataset', help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('evaluation', help="Evaluation dataset with pairs")
    parser.add_argument('-t', required=False, default=[], nargs='+', type=float, help="threshold dataset with pairs")
    parser.add_argument('-k', nargs='+', type=int, help="threshold dataset with pairs")
    parser.add_argument('-w', help="Time window")
    parser.add_argument('-add_cand', action="store_true", help="Add candidate that are not in the recommendation list")
    parser.add_argument('-result_file', help="Time window")

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

    # logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
    eval_strategy = SunStrategy(report_db, validationDataset, args.w)
    recommendation_file = RecommendationFile.create(args.result_file)

    listeners = [
        MAP_RecallRate(report_db, args.k, True),
        BinaryPredictionROC(report_db, eval_strategy.get_report_set()),
    ]

    recommendation_fixer = RecommendationFixer(eval_strategy, args.add_cand)
    logger.info("Start iteration")
    recommendation_file.iterate(eval_strategy, listeners, recommendation_fixer)
    logger.info("Result iteration is over")


    for l in listeners:
        obj = l.compute()

        if isinstance(obj, list):
            for o in obj:
                logger.info(o)
        else:
            logger.info(obj)
