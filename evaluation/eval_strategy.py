import json
import logging
import os
import struct
from itertools import zip_longest, islice
from time import time

import h5py
import numpy as np

from util.data_util import read_date_from_report


def log_and_reset_metric(logger, recommendation_file, strategy, listeners, fixer):
    recommendation_file.iterate(strategy, listeners, fixer)

    for l in listeners:
        obj = l.compute()

        if isinstance(obj, list):
            for o in obj:
                logger.info(o)
        else:
            logger.info(obj)

        l.reset()


def generate_recommendation_list(recommendation_path, eval_strategy, scorer, sparse, mem_only=False, max_length=-1,
                                 reuse=0):
    n_reports = len(eval_strategy)

    if recommendation_path is None or len(recommendation_path) == 0:
        logging.getLogger().info("RecommendationMem")
        recommendation_file = RecommendationMem(eval_strategy.get_report_db(), n_reports)
    elif sparse:
        if reuse > 0:
            logging.getLogger().info("Reusing file sparse")
            old_recommendation_gen = RecommendationFileSparse(eval_strategy.get_report_db(), recommendation_path,
                                                              n_reports, only_read=True).read_file()

        logging.getLogger().info("RecommendationFileSparse")
        recommendation_file = RecommendationFileSparse(eval_strategy.get_report_db(), recommendation_path, n_reports)
    else:
        logging.getLogger().info("RecommendationFileH5")
        recommendation_file = RecommendationFileH5(recommendation_path, n_reports, max_length=max_length)

    scorer.preprocess_reports(eval_strategy.get_report_set(), eval_strategy.get_queries())

    logger = logging.getLogger()
    start_time = time()

    idx = 0
    for query_id in eval_strategy:
        if eval_strategy.is_to_ignore(query_id):
            scorer.add_report(query_id)
            continue

        if reuse > 0:
            q_id, candidates, scores = next(old_recommendation_gen)

            if q_id != query_id:
                raise Exception("Query ids are different")

            recommendation_list = zip(candidates, scores)
            reuse -= 1
        else:
            candidates = eval_strategy.get_candidate_list(query_id)

            if idx > 0 and idx % 500 == 0:
                logger.info('It generated recommendation of {} reports'.format(idx))

            if len(candidates) == 0:
                if isinstance(eval_strategy, SunStrategy):
                    # If the window of days is too small to contain a duplicate bug, so this can happen.
                    logging.getLogger().warning("Bug {} has 0 candidates!".format(query_id))
                else:
                    # This shouldn't happen with the other methodologies
                    raise Exception("Bug {} has 0 candidates!".format(query_id))

                recommendation_list = candidates
            else:
                scores = scorer.score(query_id, candidates)

                # Remove pair (duplicateBug, duplicateBug) and create tuples with bug id and its similarity score.
                filtered_list = filter(lambda b: b[0] != query_id, zip(candidates, scores))

                # Sort  in descending order the bugs by probability of being duplicate
                recommendation_list = sorted(filtered_list, key=lambda x: (x[1], x[0]), reverse=True)

                # scores = [(cand_id, score) for cand_id, score in zip(candidates, scores) if cand_id != query_id]
                # recommnedation_list = sorted(scores, key=lambda x: x[1], reverse=True)

        recommendation_file.add(query_id, recommendation_list)
        recommendation_file.flush()

        del recommendation_list
        idx += 1

        # logger.info("Time to generate recommendation of {} reports: {}".format(query_id, (time() - start_bug)))

    logger.info("Time to generate recommendation of {} reports: {}".format(n_reports, (time() - start_time)))
    recommendation_file.flush()

    return recommendation_file


class RecommendationFixer(object):

    def __init__(self, strategy, add_cand):
        self.strategy = strategy
        self.add_cand = add_cand

    def fix(self, report_id, candidates, scores):
        correct_candidates = set(self.strategy.get_candidate_list(report_id))

        new_candidates = []
        new_scores = []

        if len(candidates) > len(correct_candidates):
            raise Exception("recommendation list of {} contains invalid candidates".format(report_id))

        for candidate_id, scores in zip(candidates, scores):
            new_candidates.append(candidate_id)
            new_scores.append(scores)

        if len(candidates) < len(correct_candidates):
            missing_candidates = correct_candidates - set(candidates)

            if not self.add_cand:
                raise Exception("Candidates {} are out of recommendation list".format(missing_candidates))

            for candidate_id in sorted(missing_candidates, reverse=True):
                new_candidates.append(candidate_id)
                new_scores.append(-999999999.99)

        # Check candidates list
        l1 = sorted(correct_candidates)
        l2 = sorted(new_candidates)

        for c1, c2 in zip_longest(l1, l2):
            if c1 != c2:
                raise Exception(
                    "Candidate list of {} is incorrect.\n\tincorrect:{}\n\tcorrect:{}".format(report_id, l2,
                                                                                              l1))
        return new_candidates, new_scores

    def flush(self):
        pass


class RecommendationFile(object):

    @staticmethod
    def create(path):
        ext = os.path.splitext(path)[1]

        if ext == '.json':
            return RecommendationFileJson(path)
        elif ext == '.bin':
            return RecommendationFileBin(path)
        elif ext == '.sparse':
            return RecommendationFileSparse(None, path, -1, only_read=True)
        else:
            return RecommendationFileH5(path, None, True)

    def __iter__(self):
        raise NotImplementedError()

    def iterate(self, strategy, listeners, fixer):
        queries = set(strategy.get_queries())
        tested_queries = set()

        for report_id, candidates, scores in self:
            if strategy.is_to_ignore(report_id):
                continue

            # Fix the candidate list.
            candidates, scores = fixer.fix(report_id, candidates, scores)

            # Check if report id is a correct query
            if report_id not in queries:
                raise Exception("Query {} shouldn't be evaluated.".format(report_id))

            # add query
            tested_queries.add(report_id)

            # Check if candidate list is empty
            if len(candidates) == 0:
                logging.getLogger().warning("Bug {} has 0 candidates!".format(report_id))
                continue

            # Update listeners
            for l in listeners:
                l.update(report_id, candidates, scores)

        if len(queries - tested_queries) > 0:
            for t in (queries - tested_queries):
                if not strategy.is_to_ignore(t):
                    raise Exception("Missing queries: {}".format(queries - tested_queries))

    def flush(self):
        pass


class RecommendationMem(RecommendationFile):

    def __init__(self, bug_db, n_reports):
        logger = logging.getLogger(__name__)
        self.bug_db = bug_db
        self.n_queries = 0
        self.n_reports = n_reports
        self.master_id_by_id = self.bug_db.get_master_by_report()
        self.recommendation_list = []

    def __iter__(self):
        return iter(self.recommendation_list)

    def add(self, report_id, recommendation_list):
        report_row = []
        score_row = []

        for cand_id, score in recommendation_list:
            report_row.append(cand_id)
            score_row.append(score)

        self.recommendations.append((report_id, report_row, score_row))

    def add(self, report_id, recommendation_list):
        if self.n_queries >= self.n_reports:
            logging.getLogger().warning(f"More queries than expected: {self.n_queries}")

        master_id = self.master_id_by_id[report_id]
        is_dup = master_id != report_id
        reports = []
        has_found = False
        n_tops = 5

        for idx, (cand_id, score) in enumerate(recommendation_list):
            reports.append((cand_id, score))

            if is_dup:
                if self.master_id_by_id[cand_id] == master_id:
                    has_found = True

                if has_found and idx + 1 >= n_tops:
                    break
            elif idx + 1 >= n_tops:
                break

        candidates = []
        scores = []

        for cand_id, score in reports:
            candidates.append(cand_id)
            scores.append(score)

        self.recommendation_list.append((report_id, candidates, scores))
        self.n_queries += 1


class RecommendationFileH5(RecommendationFile):

    def __init__(self, filepath, n_reports, only_read=False, max_length=-1):
        logger = logging.getLogger(__name__)
        self.only_read = only_read
        self.max_length = max_length

        if max_length > 0:
            logger.warning("Max length was enabled ({})".format(max_length))

        if only_read:
            self.file = h5py.File(filepath, 'r')
            self.dset = self.file['recommendation_list']
        else:
            if os.path.exists(filepath):
                idx = 1
                filename, ext = os.path.splitext(filepath)
                new_file_path = "{}_{}{}".format(filename, idx, ext)
                while os.path.exists(new_file_path):
                    idx += 1
                    new_file_path = "{}_{}{}".format(filename, idx, ext)

                logger.warning("{} exists. The content will redirected to {}".format(filepath, new_file_path))
                filepath = new_file_path

            self.n_reports = n_reports
            self.file = h5py.File(filepath, 'w')

            dt = h5py.vlen_dtype(np.dtype('float32'))
            self.dset = self.file.create_dataset("recommendation_list", (n_reports, 3), dtype=dt)
            self.idx = 0

            logger.info({"Create recommendation file: {}".format(filepath)})

    def __iter__(self):
        return map(lambda row: (int(row[0][0]), row[1].astype(int), row[2]), self.dset)

    def flush(self):
        self.file.flush()

    def add(self, report_id, recommendation_list):
        report_row = []
        score_row = []

        if self.only_read:
            raise Exception("Only read!")

        if self.idx >= self.n_reports:
            raise Exception("Exceed file capacity")

        r_iter = islice(recommendation_list, self.max_length) if self.max_length > 0 else iter(recommendation_list)

        for cand_id, score in r_iter:
            report_row.append(cand_id)
            score_row.append(score)

        self.dset[self.idx] = [[report_id], report_row, score_row]
        self.idx += 1


class RecommendationFileBin(RecommendationFile):

    def __init__(self, filepath):
        self.filepath = filepath

    def read_file(self):
        with open(self.filepath, "rb") as f:
            n_queries = int.from_bytes(f.read(8), byteorder="big")

            for _ in range(n_queries):
                query_id = int.from_bytes(f.read(8), byteorder="big")
                n_scores = int.from_bytes(f.read(8), byteorder="big")

                candidates = []
                scores = []

                for _ in range(n_scores):
                    candidates.append(int.from_bytes(f.read(8), byteorder="big"))
                    scores.append(struct.unpack(">d", f.read(8))[0])
                yield (query_id, candidates, scores)

    def __iter__(self):
        return iter(self.read_file())


class RecommendationFileSparse(RecommendationFile):

    def __init__(self, bug_db, filepath, n_reports, only_read=False, n_tops=50):
        self.bug_db = bug_db
        self.filepath = filepath
        self.n_tops = n_tops
        self.n_queries = 0
        self.n_reports = n_reports
        self.only_read = only_read
        if only_read:
            self.file = None
            self.master_id_by_id = None
        else:
            self.master_id_by_id = self.bug_db.get_master_by_report()

            if os.path.exists(filepath):
                idx = 1
                filename, ext = os.path.splitext(filepath)
                new_file_path = "{}_{}{}".format(filename, idx, ext)
                while os.path.exists(new_file_path):
                    idx += 1
                    new_file_path = "{}_{}{}".format(filename, idx, ext)

                logging.getLogger().warning(
                    "{} exists. The content will redirected to {}".format(filepath, new_file_path))
                self.filepath = new_file_path

            self.file = open(self.filepath, "wb")

            self.file.write(struct.pack(">i", n_reports))
            self.file.write(struct.pack(">i", n_tops))

    def read_file(self):
        if self.file is not None:
            self.file.close()

        with open(self.filepath, "rb") as f:
            n_queries = struct.unpack(">i", f.read(4))[0]
            n_tops = struct.unpack(">i", f.read(4))[0]

            for _ in range(n_queries):
                query_id = struct.unpack(">q", f.read(8))[0]
                n_scores = struct.unpack(">i", f.read(4))[0]

                candidates = []
                scores = []

                for _ in range(n_scores):
                    candidates.append(struct.unpack(">q", f.read(8))[0])
                    scores.append(np.float32(struct.unpack(">d", f.read(8))[0]))

                yield (query_id, candidates, scores)

    def __iter__(self):
        return iter(self.read_file())

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def add(self, report_id, recommendation_list):
        if self.only_read:
            raise Exception("Only read!")

        if self.n_queries >= self.n_reports:
            raise Exception("Exceed file capacity")

        master_id = self.master_id_by_id[report_id]
        is_dup = master_id != report_id
        reports = []
        has_found = False
        for idx, (cand_id, score) in enumerate(recommendation_list):
            reports.append((cand_id, score))

            if is_dup:
                if self.master_id_by_id[cand_id] == master_id:
                    has_found = True

                if has_found and idx + 1 >= self.n_tops:
                    break
            elif idx + 1 >= self.n_tops:
                break

        self.file.write(struct.pack(">q", report_id))
        self.file.write(struct.pack(">i", len(reports)))

        for cand_id, score in reports:
            self.file.write(struct.pack(">q", cand_id))
            self.file.write(struct.pack(">d", score))

        self.n_queries += 1

        if self.n_queries == self.n_reports:
            self.file.close()
            self.file = None


class RecommendationFileJson(RecommendationFile):

    def __init__(self, filepath):
        try:
            self.results = json.load(open(filepath, 'r'))
        except:
            try:
                self.results = json.load(open(filepath, 'r'))

                for idx, (report_id, _, cand_score) in enumerate(self.results):
                    candidates = []
                    scores = []

                    for cand_id, score in cand_score:
                        candidates.append(cand_id)
                        scores.append(score)

                    self.results[idx] = (report_id, candidates, scores)
            except:
                # Old version
                self.results = []

                with open(filepath, 'r') as f:
                    for l in f:
                        obj = json.loads(l)
                        report_id = obj[0]
                        candidates = []
                        scores = []

                        for cand, s in obj[2]:
                            candidates.append(cand)
                            scores.append(s)

                        self.results.append((report_id, candidates, scores))

    def __iter__(self):
        return iter(self.results)


class SunStrategy(object):

    def __init__(self, report_db, dataset, window):
        self.report_db = report_db
        self.master_id_by_report_id = self.report_db.get_master_by_report()
        self.queries = dataset.bugIds
        self.duplicate_reports = dataset.duplicateIds
        self.candidates = []
        self.window = int(window) if window is not None else 0
        self.creation_dates_by_master_id = {}
        self.logger = logging.getLogger()

        # Get oldest and newest duplicate bug report in dataset
        newest_report = (
            self.queries[0], read_date_from_report(self.report_db.get_report(self.queries[0])))

        for report_id in self.queries:
            dup = self.report_db.get_report(report_id)
            creation_date = read_date_from_report(dup)

            if newest_report[1] < creation_date or (newest_report[1] == creation_date and report_id > newest_report[0]):
                newest_report = (report_id, creation_date)

        # Keep only reports that were created before the newest report in the dataset
        for report in self.report_db.report_list:
            creation_date = read_date_from_report(report)
            report_id = report['bug_id']

            if not isinstance(report_id, int):
                raise Exception("Report id has to be a number")

            # Remove bugs that their creation time is bigger than newest duplicate bug
            if creation_date > newest_report[1] or (
                    creation_date == newest_report[1] and report['bug_id'] > newest_report[0]):
                continue

            self.candidates.append((report_id, creation_date.timestamp(), self.master_id_by_report_id[report_id]))

        # Store all creation date in each master set
        for masterId, masterSet in self.report_db.get_master_set_by_id(
                map(lambda c: c[0], self.candidates)).items():
            ts_list = []

            for report_id in masterSet:
                if masterId > report_id:
                    raise Exception("MasterId {} is greater than duplicate {}".format(masterId, report_id))

                bug_creation_date = read_date_from_report(self.report_db.get_report(report_id))
                ts_list.append((int(report_id), bug_creation_date.timestamp()))

            self.creation_dates_by_master_id[masterId] = ts_list

        # Set all reports that are going to be used by our models.
        self.report_set = set(map(lambda x: x[0], self.candidates))
        self.report_set.update(self.queries)

    def get_report_db(self):
        return self.report_db

    def get_queries(self):
        return self.queries

    def get_report_set(self):
        return self.report_set

    def get_filtered_report_set(self):
        raise NotImplementedError()

    def get_duplicate(self):
        return self.duplicate_reports

    def __iter__(self):
        return iter(self.queries)

    def __len__(self):
        return len(self.queries)

    def is_grouped_by_master(self):
        # Return whether the reports are grouped by their master set.
        return True

    def is_to_ignore(self, query_id):
        return not self.report_db[query_id].get("label", True)

    def get_candidate_list(self, query_id):
        query = self.report_db.get_report(query_id)
        query_creation_date = read_date_from_report(query)
        query_timestamp = query_creation_date.timestamp()
        query_day_timestamp = int(query_timestamp / (24 * 60 * 60))

        master2timestamp = {}

        for master_id, ts_master_set in self.creation_dates_by_master_id.items():
            bucket_timestamp = 0

            for report_id, ts in ts_master_set:
                # Ignore reports that were created after the query
                if ts > query_timestamp or (ts == query_timestamp and report_id >= query_id):
                    continue

                # Get newest ones
                if bucket_timestamp < ts:
                    bucket_timestamp = ts

            master2timestamp[master_id] = bucket_timestamp

        candidates = []
        n_skipped = 0

        for candidate_id, candidate_timestamp, cand_master_id in self.candidates:
            # Ignore reports that were created after the anchor report
            if candidate_timestamp > query_timestamp or (
                    candidate_timestamp == query_timestamp and candidate_id > query_id):
                continue

            # Check if the same report
            if candidate_id == query_id:
                continue

            bucket_timestamp = master2timestamp.get(cand_master_id, candidate_timestamp)

            # Transform to day timestamp
            report_day_timestamp = int(bucket_timestamp / (24 * 60 * 60))

            # Is it in the window?
            if 0 < self.window < (query_day_timestamp - report_day_timestamp):
                n_skipped += 1
                continue

            # It is a candidate
            candidates.append(candidate_id)

        return candidates