import argparse
import json
import logging
from types import SimpleNamespace

from hyperopt import fmin, tpe, STATUS_OK, Trials

from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset
from experiments import trace_sim, brodie_05, rebucket, edit_distance, prefix_match, moroo
from util.jsontools import JsonLogFormatter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bug_dataset', help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('evaluation', help="Evaluation dataset with pairs")
    parser.add_argument('method', help="")
    parser.add_argument('search_space_script', help="")
    parser.add_argument('-max_evals', default=100, type=float, help="threshold dataset with pairs")

    parser.add_argument('-test')
    parser.add_argument('-t', nargs='+', type=float, help="threshold dataset with pairs")
    parser.add_argument('-k', nargs='+', type=int, help="Recall rate @k")
    parser.add_argument('-w', help="Time window")
    parser.add_argument('-df_threshold', default=0.0, type=float, help="threshold dataset with pairs")
    parser.add_argument('-field_coef', default=0.0, type=float, help="")
    parser.add_argument('-result_file', help="")
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
    parser.add_argument('-top_n_file_validation', help="")
    parser.add_argument('-top_n_file_test', help="")

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

    f = open(args.search_space_script, 'r')
    code_str = f.read()
    f.close()

    fixed_values = {}

    exec(compile(code_str, args.search_space_script, 'exec'))

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


    def objective(x):
        std_args = vars(args)

        for k, v in fixed_values.items():
            std_args[k] = v

        for k, v in x.items():
            std_args[k] = v

        std_args["top_n_file"] = args.top_n_file_validation

        results = method_func(SimpleNamespace(**std_args), report_db, validationDataset)

        for r in results:
            if r['label'] == 'MAP_RecallRate':
                map = r['map']
            elif r['label'] == 'BinaryPredictionROC':
                auc = r['auc']

        print("x={}\tloss={}\tmap={}\tauc={}args={}".format(x, 2.0 - (map + auc), map, auc, std_args))
        return {'loss': 2.0 - (map + auc), 'status': STATUS_OK, 'results': results, 'map': map, 'auc': auc,
                'args': dict(std_args)}


    trials = Trials()
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.max_evals,
                trials=trials)

    sorted_results = sorted(trials, key=lambda k: k['result']['loss'], reverse=True)
    for t in sorted_results:
        result = t['result']
        print("vals={}\tloss={}\tmap={}\tauc={}\tresult={}\targs={}".format(t['misc']['vals'], result['loss'],
                                                                            result['map'], result['auc'],
                                                                            result['results'], result['args']))

    print(best)
    best_result = sorted_results[-1]

    if args.test is not None:
        testDataset = ReportDataset(args.test)
        std_args = best_result['result']['args']
        std_args['evaluation'] = testDataset
        std_args["top_n_file"] = args.top_n_file_test

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
