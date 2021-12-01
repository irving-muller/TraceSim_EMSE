"""
It extracts stack traces in the description and attachments of bug reports.
We use the same extractor Parse::StackTrace used in Gnome

To use this script, you have to install perl and cpan. Also, run "cpan Parse::StackTrace" to install the module Parse::StackTrace.
"""

import argparse
import codecs
import hashlib
import logging
import re
import json
from datetime import timedelta, datetime, timezone
from itertools import zip_longest, islice
from subprocess import Popen, PIPE
from time import time

import pymongo


class CStackTraceExtractor(object):

    @staticmethod
    def standardize(stacktrace_json):
        stacktraces = []

        for thread in stacktrace_json.get('threads', []):
            frames = []

            for frame in thread['frames']:
                function = frame['function']

                if function == '??':
                    function = None
                elif re.match(r"0x[0-9a-zA-Z]+ in", function) is not None:
                    function = None

                o = {
                    'depth': frame['number'],
                    'function': function
                }

                if 'library' in frame:
                    o['dylib'] = frame['library']

                if 'memory_location' in frame:
                    o['address'] = frame['memory_location']

                if 'lib_tag' in frame:
                    # o['dylib'] = frame['lib_tag']
                    o['lib_tag'] = frame['lib_tag']

                if 'args' in frame:
                    o['args'] = frame['args']

                if 'file' in frame:
                    o['file'] = frame['file']

                if 'line' in frame:
                    o['fileline'] = frame['line']

                if 'is_crash' in frame:
                    o['is_crash'] = frame['is_crash']

                if 'code' in frame:
                    o['code'] = frame['code']

                frames.append(o)

            if len(frames) == 0:
                continue

            stacktraces.append({
                "exception": None,
                "message": None,
                "frames": frames,
                "nested": [],
            })

        return stacktraces

    def extract_from_txt(self, text, report_id):
        stacktraces = []

        if not isinstance(text, str):
            logging.warning("The description of bug {} is not a string {}".format(report_id, text))
            return stacktraces

        m = re.search(r"#[0-9]+( +0[xX][0-9a-zA-Z]+)? +(in +)?[^()\n]+\([^)]*\)( +(from|at) [0-9a-zA-Z/.:-]+)?", text)

        a = time()
        p = Popen(["perl", "./extract_stack_trace.pl"], stdin=PIPE, stdout=PIPE)
        outs, errs = p.communicate(input=text.encode("UTF-8"))

        if errs is not None:
            logging.getLogger().error(errs)
            logging.getLogger().error(text)

        stack = json.loads(outs.decode("UTF-8"))
        logging.getLogger().info(time() - a)

        if len(stack) == 0:
            if m is not None:
                logging.getLogger().info(
                    "It seems that Parse did not detect an stacktrace in {}. Detected str: {}".format(report_id,
                                                                                                      m.group()))

            return []

        return self.standardize(stack, report_id)

    def extract(self, batch):
        texts = []
        indexes = []

        for idx, doc in enumerate(batch):
            texts.append(doc['description'])
            indexes.append((idx, None))

            for att_idx, att in enumerate(doc['attachments']):
                texts.append(att['data'])
                indexes.append((idx, att_idx))

        # m = re.search(r"#[0-9]+( +0[xX][0-9a-zA-Z]+)? +(in +)?[^()\n]+\([^)]*\)( +(from|at) [0-9a-zA-Z/.:-]+)?", text)
        json_in = json.dumps(texts)

        # a = time()
        p = Popen(["perl", "./extract_stack_trace.pl"], stdin=PIPE, stdout=PIPE)
        outs, errs = p.communicate(input=json_in.encode("UTF-8"))

        if errs is not None:
            logging.getLogger().error(errs)

        json_out = json.loads(outs.decode("UTF-8"))
        # logging.getLogger().info(time() - a)

        if len(indexes) != len(json_out):
            logging.getLogger().error(json_in)
            raise Exception("The number of texts and perl output is different!!")

        for (list_idx, att_idx), st in zip_longest(indexes, json_out):
            doc = batch[list_idx]

            st = [] if st is None else CStackTraceExtractor.standardize(st)

            if att_idx is None:
                doc['stacktrace'] = st
            else:
                att = doc['attachments'][att_idx]
                att['stacktrace'] = st

        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-db', required=True, help="Mongo database")
    parser.add_argument('-c', required=True, help="Mongo collection")
    parser.add_argument('-output', required=True, help="Mongo collection")
    parser.add_argument('--no_tree', action="store_true")
    parser.add_argument('-host', default="127.0.0.1", help="Mongo collection")

    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    args = parser.parse_args()
    logger.info(args)

    # Connect to mongo dataset
    client = pymongo.MongoClient(args.host)
    database = client[args.db]
    col = database[args.c]

    """
    Bug reports to test:
        Bug 2755: at org.eclipse.swt.widgets.EventTable.sendEvent(EventTable.java(Compiled Code))
        Bug 5791: nested stack trace
        Bugs 98298, 98918, 99010, 103656, 116964, 121060, 121742: the word "at" in the method name 
        Bug 397666: <clinit> and <init>
    """
    #
    bug_reports = []
    extractor = CStackTraceExtractor()

    docs = []

    bucket_by_id = {}
    dup_id_by_id = {}
    bugs = []

    for doc in col.find({}):
        if isinstance(doc['dup_id'], list) and len(doc['dup_id']) == 0:
            doc['dup_id'] = None

        if doc['dup_id'] is not None and doc['dup_id']:
            dup_id_by_id[doc['bug_id']] = doc['dup_id']

            l = bucket_by_id.setdefault(doc['dup_id'], [])
            l.append(doc)

        bugs.append(doc)

    if args.no_tree:
        # Remove cycle
        for bug_id in list(bucket_by_id.keys()):
            dup_id = dup_id_by_id.get(bug_id)

            if dup_id is not None and dup_id_by_id.get(dup_id) == bug_id:
                del dup_id_by_id[bug_id]

                for i, b in enumerate(bucket_by_id[dup_id]):
                    if b['bug_id'] == bug_id:
                        b['dup_id'] = None
                        del bucket_by_id[dup_id][i]
                        break

                if len(bucket_by_id[dup_id]) == 0:
                    del bucket_by_id[dup_id]

        # Remove recurrency
        old_dup_id_by_id = dict(dup_id_by_id)

        for bug_id, bucket in bucket_by_id.items():
            dup_id = dup_id_by_id.get(bug_id)

            if dup_id is not None:
                for b in bucket:
                    b['dup_id'] = dup_id
                    dup_id_by_id[b['bug_id']] = dup_id

                bucket_by_id[dup_id] = bucket_by_id[dup_id] + bucket

        # Check no tree operations
        for doc in bugs:
            if doc['dup_id'] is None:
                continue
            old_dup_id = old_dup_id_by_id[doc['bug_id']]

            if doc['dup_id'] == old_dup_id:
                continue

            if doc['dup_id'] != dup_id_by_id[old_dup_id] or doc['dup_id'] in dup_id_by_id:
                raise Exception(doc)
    # 10 minutes
    threshold_att = timedelta(seconds=600)
    batch_size = 500

    iterator = iter(bugs)

    while True:
        batch = list(islice(iterator, batch_size))

        if len(batch) == 0:
            break

        for doc in extractor.extract(batch):
            stacktraces = doc['stacktrace']
            submission_date = doc['creation_ts']

            if len(doc['attachments']) > 0:
                # Get the first file with stacktrace that was submitted within  threshold_att seconds. File were filtered during the report retrieval.
                for att in sorted(doc['attachments'], key=lambda att: att['date']):
                    att_st = att['stacktrace']

                    if len(att_st) == 0:
                        continue

                    logging.getLogger().info("We found an stacktrace {} in Report {}".format(att['id'], doc['bug_id']))
                    if att["date"] - submission_date <= threshold_att:
                        stacktraces.extend(att_st)

                    break

            # Remove duplicate stack traces
            st_set = set()
            unique_st = []

            for st in stacktraces:
                st_hash = hashlib.sha512(json.dumps(st).encode("utf-8")).hexdigest()

                if st_hash in st_set:
                    continue

                st_set.add(st_hash)
                unique_st.append(st)

            if len(unique_st) == 0:
                continue

            logging.getLogger().info("We found {} stacktraces in Report {}".format(len(unique_st), doc['bug_id']))

            l = [len(st['frames']) for st in unique_st]

            if len(l) > 0 and sum(map(lambda k: int(k > 3), l)) == 0:
                logging.warning("All stack lengths are smaller than 3. BugId:{},{}".format(doc['bug_id'], l))
                a = 1
                a += 2

            doc['stacktrace'] = unique_st
            docs.append(doc)

            del doc['_id']


    def datetime2str(obj):
        if isinstance(obj, datetime):
            # UTC time
            # https://docs.python.org/3/library/datetime.html
            return obj.replace(tzinfo=timezone.utc).timestamp()


    with codecs.open(args.output, 'w') as f:
        json.dump(docs, f, default=datetime2str)
