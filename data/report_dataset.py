"""
Each dataset has bug report ids and the ids of duplicate bug reports.
"""
class ReportDataset(object):

    def __init__(self, file,bug_Ids=None,duplicateIds=None):
        if file is not None:
            f = open(file, 'r')
            self.info = f.readline().strip()

            self.bugIds = [int(id) for id in f.readline().strip().split()]
            self.duplicateIds = [int(id) for id in f.readline().strip().split()]
        else:
            self.bugIds = bug_Ids
            self.duplicateIds = duplicateIds