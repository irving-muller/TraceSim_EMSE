import argparse
import json
import logging
from types import SimpleNamespace

import numpy
import time

from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset
from evaluation.metric import MAP_RecallRate
from util.jsontools import JsonLogFormatter

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from experiments import trace_sim, brodie_05, rebucket, edit_distance, prefix_match, moroo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bug_dataset', help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('test', help="Evaluation dataset with pairs")
    parser.add_argument('method', help="")
    parser.add_argument('arg_files', help="")
    parser.add_argument('chunk_id', help="")
    parser.add_argument('-top_n_file', help="")


    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logHandler = logging.StreamHandler()
    formatter = JsonLogFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.info(args)


    if args.method == "brodie_05":
        method_func = brodie_05.run
    elif args.method == "damerau_levenshtein":
        method_func = edit_distance.run_damerau
    elif args.method == "opt_align":
        method_func = edit_distance.run_opt_align
    elif args.method == "pdm":
        method_func = rebucket.run
    elif args.method == "trace_sim":
        method_func = trace_sim.run
    elif args.method == "prefix_match":
        method_func = prefix_match.run
    elif args.method == "moroo":
        method_func = moroo.run

    args_by_chunk_id = json.load(open(args.arg_files))

    report_db = BugReportDatabase.from_json(args.bug_dataset)
    testDataset = ReportDataset(args.test)

    std_args = args_by_chunk_id[args.chunk_id]
    std_args['evaluation'] = args.test
    std_args['top_n_file'] = args.top_n_file

    nm = SimpleNamespace(**std_args)
    print("#### Evaluating on Test")
    print(nm)
    results = method_func(nm, report_db, testDataset)

    for r in results:
        if r['label'] == 'MAP_RecallRate':
            map = r['map']
        elif r['label'] == 'BinaryPredictionROC':
            auc = r['auc']

    print(json.dumps(results))
