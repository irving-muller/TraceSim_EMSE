import logging
import math

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve


def fix_predicted_master_id(queries, predicted_by_report_id, master_id_by_bug_id):
    queries = set(queries)
    fixed_dict = {}

    for report_id, predicted_report in predicted_by_report_id.items():
        """
        Given a predicted report, it links the report_id to a bucket.
    
        1st case:  report_id is non duplicate (predicted report == report_id) 
        2nd case:  predicted report is in the evaludation set, i.e., the model can predict a different bucket 
            to the predicted report. 
        3rd case: predicted report is not in the evaluation set. Use the correct from the dataset.
        """
        previous_report_id = report_id
        current_report_id = predicted_report

        while current_report_id != previous_report_id:
            previous_report_id = current_report_id

            if current_report_id in queries:
                current_report_id = predicted_by_report_id[current_report_id]
            else:
                # This is the final bucket
                current_report_id = master_id_by_bug_id[current_report_id]
                break

        fixed_dict[report_id] = current_report_id

    return fixed_dict


class MAP_RecallRate(object):

    def __init__(self, report_db, k=None, group_by_master=True, label=None):
        self.master_set_by_id = report_db.get_master_set_by_id()
        self.master_id_by_bug_id = report_db.get_master_by_report()
        self.label = "MAP_RecallRate" if label is None else label

        if k is None:
            k = list(range(1, 21))

        self.k = sorted(k)

        self.sum_map = 0.0
        self.hits_per_k = dict((k, 0) for k in self.k)
        self.n_duplicate = 0

        self.logger = logging.getLogger()
        self.group_by_master = group_by_master

    def reset(self):
        self.hits_per_k = dict((k, 0) for k in self.k)
        self.n_duplicate = 0
        self.sum_map = 0.0

    def update(self, report_id, candidates, scores):
        masterset_id = self.master_id_by_bug_id[report_id]

        if masterset_id == report_id:
            # It is not duplicate
            return

        masterSet = self.master_set_by_id[masterset_id]

        pos = math.inf
        seen_masters = set() if self.group_by_master else list()

        for cand_id in candidates:
            masterset_id = self.master_id_by_bug_id[cand_id]

            if self.group_by_master:
                if masterset_id in seen_masters:
                    continue

                seen_masters.add(masterset_id)
            else:
                seen_masters.append(cand_id)

            if cand_id in masterSet:
                pos = len(seen_masters)
                # print("{}, {}, {}, {}, {}".format(report_id, pos, cand_id, self.master_id_by_bug_id[report_id], list(
                #     zip(candidates[:30], map(lambda x: self.master_id_by_bug_id[x], candidates[:30]), scores[:30]))))
                correct_cand = cand_id
                break

        # If one of k duplicate bugs is in the list of duplicates, so we count as hit. We calculate the hit for each different k
        for idx, k in enumerate(self.k):
            if k < pos:
                continue

            self.hits_per_k[k] += 1

        self.sum_map += 1 / pos
        self.n_duplicate += 1

        return pos

    def compute(self):
        recall_rate = {}

        for k, hit in self.hits_per_k.items():
            rate = float(hit) / self.n_duplicate
            recall_rate[k] = rate

        return {'type': "metric",
                'label': self.label,
                'hits_per_k': self.hits_per_k,
                "rr": recall_rate,
                "sum_map": self.sum_map,
                "map": self.sum_map / self.n_duplicate,
                "total": self.n_duplicate,
                }


class BinaryPredictionROC(object):

    def __init__(self, report_db, all_reports, label=None):
        self.all_reports = list(all_reports)
        self.master_id_by_bug_id = report_db.get_master_by_report(self.all_reports)
        self.master_reports = list(set((self.master_id_by_bug_id[report_id] for report_id in self.all_reports)))
        self.label = "BinaryPredictionROC" if label is None else label

        self.queries = []
        self.scores = []

    def reset(self):
        self.scores = []
        self.queries = []

    def update(self, query_id, candidates, scores):
        top_candidate = int(candidates[0])
        top_score = scores[0]

        self.queries.append(query_id)
        self.scores.append(top_score)
        # print("{} {} {} {}".format(query_id, top_candidate, top_score, self.master_id_by_bug_id[query_id] != query_id))

        # print("{}, {}, {}, {}, {}, {}, {}".format(
        #     self.master_id_by_bug_id[query_id] == query_id, query_id,
        #     self.master_id_by_bug_id[query_id], top_candidate, self.master_id_by_bug_id[top_candidate], top_score,
        #     [(k, s) for k, s in zip(candidates[:20], scores[:20])]))

    def compute(self):
        y_true = [int(self.master_id_by_bug_id[report_id] != report_id) for report_id in self.queries]

        fpr, tpr, thresholds = roc_curve(y_true, self.scores, pos_label=1)

        return {'type': "metric",
                'label': self.label,
                'fpr': fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": metrics.auc(fpr,tpr),
                "n_queries": len(y_true),
                "threshold": thresholds.tolist(),
                }

