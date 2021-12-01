"""
It extracts stack traces in the description of bug reports.
We implemented the same regular expression in "Finding Duplicates of Your Yet Unwritten Bug Report. Johannes Lerch. 2013"
    that capture stack traces from JAVA.
"""

import argparse
import codecs
import hashlib
import logging
import re
import json
from datetime import timedelta, datetime, timezone

import pymongo


class StackTraceExtractor(object):
    """
    We implemented the same regular expression in "Finding Duplicates of Your Yet Unwritten Bug Report. Johannes Lerch. 2013"
    - It is required that the first module of fully qualified method does not contain any space.
        This allows that the regex do not consider the word "at" in the message as the beginning of  a method name.
            Bug reports with "at" in the message: 98298, 98918, 99010, 103656, 116964, 121060, 121742
    - The Exception class name can be Error or Exception. Bug 170495
    """

    exception = "(([\w]+([.$/][\w]*)+)(Exception|Error))"
    message = ".*?"
    exception_message = "%s( *?[:] (%s))?" % (exception, message)

    # org.netbeans.modules.form.fakepeer.$Proxy26.setZOrder
    method = "[\w$]+([.$/]\$?[\w ]+)*([.$/](<init>|<clinit>))?"
    source = ".*?"
    frames = "( +?at +(%s) *?[(]{1,3}(%s)[)]{1,3})" % (method, source)

    stacktrace_regex = r'%s%s+' % (exception_message, frames)
    regex = '%s( +caused +by *: +%s)?' % (stacktrace_regex, stacktrace_regex)

    exception_msg_re = re.compile(exception_message + r" +?at [\w$]+[.$/]", re.IGNORECASE)
    func_call_re = re.compile(frames, re.IGNORECASE)
    compiled_re = re.compile(regex, re.IGNORECASE)

    def extract_elements(self, stacktrace, bug_id):
        exp_mst_sre = self.exception_msg_re.search(stacktrace)

        try:
            exception_content = exp_mst_sre.group(1)
            msg_content = exp_mst_sre.group(6)
        except:
            logging.warning(
                "It was not possible to extract the exception and msg from the stack trace. It is probably a false positive.BugId:{}\t{}".format(
                    bug_id,
                    stacktrace))
            return None

        frames = []

        for func_call_sre in self.func_call_re.finditer(stacktrace):
            method_content = func_call_sre.group(2)
            source_content = func_call_sre.group(6)

            source_split = source_content.rsplit(':', 1)

            if len(source_split) == 1:
                #  Compiled code or Native method
                filename = source_content
                fileline = None
            else:
                filename, fileline = source_content.rsplit(':', 1)

                fileline = re.sub('[^0-9]', '', fileline)

                if len(fileline) == 0:
                    fileline = None
                else:
                    fileline = int(fileline)


            # One allowed white space per 20 characters while multiple consecutive white spaces are counted as one
            # This removes incomplete stack traces luke bug report 56279 and 67464.
            method_content = method_content.strip()

            if method_content.count(' ') / len(method_content) > 0.05:
                logging.warning(
                    "Fully qualified method name contains more than 1 space by 20 characters. BugId:{}\t{}".format(
                        bug_id, method_content))
                logging.warning("Discarding stack trace.BugId:{}\t{}".format(bug_id, stacktrace))
                return None

            method_content = method_content.replace(' ', '')
            filename = filename.replace(' ','')

            frames.append((method_content, filename, fileline))

        return {
            "exception": exception_content,
            "message": msg_content,
            "frames": [{"function": fc[0], "file": fc[1], "fileline": fc[2], "depth": depth} for depth, fc in
                       enumerate(frames)]
        }

    def extract(self, text, bug_id):
        stacktraces = []
        text = re.sub('\[[a-zA-Z]+\]', '', text)

        if not isinstance(text, str):
            logging.warning("The description of bug {} is not a string {}".format(bug_id, text))
            return stacktraces

        text = re.sub(r'\s+', ' ', text)

        for sre in self.compiled_re.finditer(text):
            stacktrace = sre.group()
            split_stacktrace = re.split("caused +by *: +", stacktrace, flags=re.IGNORECASE)

            main_stacktrace = split_stacktrace[0]
            main_stacktrace = self.extract_elements(main_stacktrace, bug_id)

            if main_stacktrace is None:
                continue

            nested_stacktraces = []
            for i in range(1, len(split_stacktrace)):
                nested_trace = self.extract_elements(split_stacktrace[i], bug_id)

                if nested_trace is None:
                    logging.warning("Remove malformed nested stack trace:{}\t{}\t".format(bug_id, split_stacktrace[i]))
                    continue

                nested_stacktraces.append(nested_trace)

            if len(split_stacktrace) > 1 and len(nested_stacktraces) == 0:
                logging.warning(
                    "Ignoring stack trace because all nested stack trace are malformed. BugId:{}\t{}".format(
                        bug_id, stacktrace))
                continue

            main_stacktrace['nested'] = nested_stacktraces
            stacktraces.append(main_stacktrace)


        return stacktraces


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
    extractor = StackTraceExtractor()

    docs = []

    logger.info(extractor.regex)
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

    # for doc in filter(lambda d: d['bug_id'] == 547516, bugs):
    for doc in bugs:
        desc = doc.get('description', '')
        stacktraces = extractor.extract(desc, doc['bug_id'])
        submission_date = doc['creation_ts']

        if len(doc['attachments']) > 0:
            # Get the first file with stacktrace that was submitted within  threshold_att seconds. File were filtered during the report retrieval.
            first_att = sorted(doc['attachments'], key=lambda att: att['date'])[0]

            if first_att["date"] - submission_date <= threshold_att:
                att_st = extractor.extract(first_att['data'], doc['bug_id'])
                stacktraces.extend(att_st)

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

        l = [len(st['frames']) for st in unique_st]

        if len(l) > 0 and sum(map(lambda k: int(k > 3), l)) == 0:
            logging.warning("All stack lengths are smaller than 3. BugId:{},{}".format(doc['bug_id'], l))
            a=1
            a+=2

        doc['stacktrace'] = unique_st
        docs.append(doc)

        del doc['_id']

    def datetime2str(obj):
        if isinstance(obj,datetime):
            # UTC time
            # https://docs.python.org/3/library/datetime.html
            return obj.replace(tzinfo=timezone.utc).timestamp()


    with codecs.open(args.output, 'w') as f:
        json.dump(docs, f,default=datetime2str)
