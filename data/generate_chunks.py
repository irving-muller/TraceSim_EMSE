import argparse
import logging
import math
import os
import random
from datetime import datetime, timezone

from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset
from util.data_util import read_date_from_report


def filter_report(query_id, query_ts, report_and_date):
    for cand_id, cand_ts in report_and_date:
        if cand_ts > query_ts or (cand_ts == query_ts and cand_id > query_id):
            continue

        yield cand_id, cand_ts

def saveDatasetFile(path, info, reports, duplicate_reports):
    f = open(path, 'w')

    f.write(info)
    f.write('\n')

    for report_id in reports:
        f.write(f'{report_id} ')

    f.write('\n')

    for dup_report_id in duplicate_reports:
        f.write(f'{dup_report_id} ')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--database', required=True, help="Json path")
    parser.add_argument('--dataset', help="")
    parser.add_argument('--n_chunks', type=int, required=True, help="")
    parser.add_argument('--folder', required=True, help="")
    parser.add_argument('--test_window', default=365, type=int)
    parser.add_argument('--n_dup_validation', default=500, type=int)
    parser.add_argument('--load_chunks')
    parser.add_argument('--seed', default=99999977377, type=int)
    parser.add_argument('--min_window_from_start', default=0, type=int)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    fileHandler = logging.FileHandler(os.path.join(args.folder, "create_dataset.log"))
    logger.addHandler(fileHandler)
    logger.info(args)

    random.seed(args.seed)

    report_db = BugReportDatabase.from_json(args.database)
    master_id_by_report_id = report_db.get_master_by_report()

    if args.dataset:
        validationDataset = ReportDataset(args.dataset)
        validation_queries = validationDataset.bugIds

        # Get oldest and newest duplicate bug report in dataset
        newest_report = (
            validation_queries[0], read_date_from_report(report_db.get_report(validation_queries[0])).timestamp())

        for report_id in validation_queries:
            dup = report_db.get_report(report_id)
            creation_date = read_date_from_report(dup).timestamp()

            if newest_report[1] < creation_date:
                newest_report = (report_id, creation_date)
    else:
        validation_queries = report_db.report_list
        logger.warning("Use all data to generate training, validation and test set")
        newest_report = (math.inf, math.inf)



    # Keep only reports that were created before the newest report in the dataset
    report_date_gen = ((report['bug_id'], read_date_from_report(report).timestamp()) for report in report_db.report_list)
    sorted_reports = sorted(filter_report(newest_report[0], newest_report[1], report_date_gen), key=lambda k: k[1])
    report_id2idx = dict(((report_id, idx) for idx, (report_id, _) in enumerate(sorted_reports)))

    start_ts_day = None

    min_window_from_start = args.min_window_from_start

    if min_window_from_start < 1:
        min_window_from_start = args.n_dup_validation


    n_dup = 0
    for report_id, report_ts in sorted_reports:
        report_day_ts = int(report_ts / (24 * 60 * 60))

        if master_id_by_report_id[report_id] == report_id:
            continue

        n_dup += 1

        if n_dup == min_window_from_start:
            start_ts_day = report_day_ts
            break

    if newest_report[1] == math.inf:
        newest_report = sorted_reports[-1]

    end_ts_day = int(newest_report[1] / (24 * 60 * 60)) - args.test_window

    if args.load_chunks is not None:
        raise NotImplementedError()
        # random_numbers = []
        #
        # for f in os.listdir(args.load_chunks):
        #     file_path = os.path.join(args.load_chunks, f)
        #
        #     if re.match(r"test_chunk_[0-9]\.txt", ) is None:
        #         continue
        #
        #     ReportDataset(file_path)
    else:
        random_numbers = random.sample(list(range(start_ts_day, end_ts_day + 1)), args.n_chunks)

    for chunk_idx, start in enumerate(random_numbers):
        end = start + args.test_window

        training_file_path = os.path.join(args.folder, "training_chunk_{}.txt".format(chunk_idx))
        validation_file_path = os.path.join(args.folder, "validation_chunk_{}.txt".format(chunk_idx))
        test_file_path = os.path.join(args.folder, "test_chunk_{}.txt".format(chunk_idx))

        test_reports = []
        test_dup = []

        n_report_b_start_end = 0

        for (report_id, report_ts) in sorted_reports:
            report_day_ts = int(report_ts / (24 * 60 * 60))

            if report_day_ts >= end:
                continue

            n_report_b_start_end+=1

            if report_day_ts < start:
                continue


            test_reports.append(report_id)

            if master_id_by_report_id[report_id] != report_id:
                test_dup.append(report_id)

        validation_reports = []
        validation_dup = []

        for idx in range(report_id2idx[test_reports[0]] - 1 , -1, -1):
            report_id, report_ts = sorted_reports[idx]
            report_day_ts = int(report_ts / (24 * 60 * 60))

            validation_reports.append(report_id)

            if start < report_day_ts:
                raise Exception("Report in validation should be in test")

            if master_id_by_report_id[report_id] != report_id:
                validation_dup.append(report_id)

            if len(validation_dup) == args.n_dup_validation:
                break

        validation_reports.reverse()
        validation_dup.reverse()


        training_reports = []
        training_dup = []

        for idx in range(report_id2idx[validation_reports[0]]):
            report_id, report_ts = sorted_reports[idx]
            training_reports.append(report_id)

            if master_id_by_report_id[report_id] != report_id:
                training_dup.append(report_id)

        start_date_str = datetime.fromtimestamp(start * (24 * 60 * 60),tz=timezone.utc).strftime("%Y/%m/%d")
        end_date_str = datetime.fromtimestamp(end * (24 * 60 * 60),tz=timezone.utc).strftime("%Y/%m/%d")

        report_set = set(test_reports)
        report_set.update(training_reports)
        report_set.update(validation_reports)

        if len(test_reports) + len(training_reports) + len(validation_reports) != len(report_set):
            raise Exception("Intersection among training, validation and test is not empty")

        dup_report_set = set(test_dup)
        dup_report_set.update(training_dup)
        dup_report_set.update(validation_dup)

        if len(test_dup) + len(training_dup) + len(validation_dup) != len(dup_report_set):
            raise Exception("Intersection among training, validation and test is not empty")

        if n_report_b_start_end != len(test_reports) + len(training_reports) + len(validation_reports):
            raise Exception("It is missing a report")

        saveDatasetFile(training_file_path, f"Chunk {chunk_idx}; {start_date_str}-{end_date_str}", training_reports,
                 training_dup)
        saveDatasetFile(validation_file_path, f"Chunk {chunk_idx}; {start_date_str}-{end_date_str}", validation_reports,
                 validation_dup)
        saveDatasetFile(test_file_path, f"Chunk {chunk_idx}; {start_date_str}-{end_date_str}", test_reports, test_dup)

        logging.info(f"Generating chunk {chunk_idx}")
        logging.info(f"\tTest_date:  {start_date_str} - {end_date_str}")
        logging.info(f"\tTraining: n_reports={len(training_reports)} n_duplicate={len(training_dup)}")
        logging.info(f"\tValidation: n_reports={len(validation_reports)} n_duplicate={len(validation_dup)}")
        logging.info(f"\tTest: n_reports={len(test_reports)} n_duplicate={len(test_dup)}")
