"""
This class represents a bug report database where we can find all bug reports that are available.
"""
import codecs
import json
import logging
from collections import OrderedDict


class BugReportDatabase(object):
    '''
    Load bug report data (categorical information, summary and description) from json file.
    '''

    def __init__(self, iterator):
        self.report_by_id = OrderedDict()
        self.report_list = []
        self.logger = logging.getLogger()

        nEmptyDescription = 0

        for report in iterator:
            if report is None:
                continue

            report_id = report["bug_id"]

            self.report_by_id[report_id] = report
            self.report_list.append(report)

    @staticmethod
    def from_json(fileToLoad):
        f = codecs.open(fileToLoad, 'r')
        return BugReportDatabase(json.load(f))

    def get_report(self, report_id):
        return self.report_by_id[report_id]

    def get_by_index(self, idx):
        return self.report_list[idx]

    def __getitem__(self, report_id):
        return self.report_by_id[report_id]

    def __iter__(self):
        return iter(self.report_list)

    def __len__(self):
        return len(self.report_list)
    
    def __contains__(self, bug):
        bugId = bug['bug_id'] if isinstance(bug, dict) else bug

        return bugId in self.report_by_id

    def get_master_by_report(self, bugs=None):
        masterIdByBugId = {}
        bugs = self.report_list if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.report_by_id[bug]

            bugId = bug['bug_id']
            dupId = bug['dup_id']

            if dupId is not None:
                masterIdByBugId[bugId] = dupId
            else:
                masterIdByBugId[bugId] = bugId

        return masterIdByBugId

    def get_master_set_by_id(self, bugs=None):
        masterSetById = {}
        bugs = self.report_list if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.report_by_id[bug]

            dupId = bug['dup_id']

            if dupId is not None:
                masterSet = masterSetById.get(dupId, set())

                if len(masterSet) == 0:
                    masterSetById[dupId] = masterSet

                masterSet.add(bug['bug_id'])

        # Insert id of the master bugs in your master sets
        for masterId, masterSet in masterSetById.items():
            if masterId in self:
                masterSet.add(masterId)

        return masterSetById
