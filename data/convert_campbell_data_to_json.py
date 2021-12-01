"""
This script extracts stack traces from txt files made available by Campbell 2016.
"""

import argparse
import codecs
import gzip
import logging
import math
import os
import re
import traceback
from datetime import datetime
from itertools import count

import sys
import json

import pymongo
import unicodedata


def fixline(line_raw, encoding_guess="utf-8"):
    line_raw = line_raw.decode(encoding=encoding_guess, errors='replace')
    line = u""
    for ch in line_raw:
        if unicodedata.category(ch)[0] == 'C':
            ch = u'?'
            # raise ValueError("Bad encoding %s in: %s" % (ch.encode('unicode_escape'), line.encode('utf-8')))
        elif ch == u'\ufffd':
            ch = u'?'
        line += ch
    return line


# number address in function (args) at file from lib
naifafl = re.compile(r'^#([\dx]+)\s+(\S+)\s+in\s+(.+?)\s+\(([^\)]*)\)\s+at\s+(\S+)\sfrom\s+(\S+)\s*$')
# number address in function (args) from lib
naifal = re.compile(r'^#([\dx]+)\s+(\S+)\s+in\s+(.+?)\s+\(([^\)]*)\)\s+from\s+(.+?)\s*$')
# number address function (args) from lib (missing in)
nafal = re.compile(r'^#([\dx]+)\s+(\S+)\s+(.+?)\s+\(([^\)]*)\)\s+from\s+(.+?)\s*$')
# number address in function (args) at file
naifaf = re.compile(r'^#([\dx]+)\s+(\S+)\s+in\s+(.+?)\s+\((.*?)\)\s+at\s+(.+?)\s*$')
# number function (args) at file
nfaf = re.compile(r'^#([\dx]+)\s+(.+?)\s+\((.*?)\)\s+at\s+(\S+)\s*$')
# number address in function (args
naifa = re.compile(r'^#([\dx]+)\s+(\S+)\s+in\s+(.+?)\s*\((.*?)\)?\s*$')
# number address in function
naif = re.compile(r'^#([\dx]+)\s+(\S+)\s+in\s+(.+?)\s*$')
# number function (args
nfa = re.compile(r'^#([\dx]+)\s+(.+?)\s+\((.*?)\)?\s*$')
# number <function>
nf = re.compile(r'^#(\d+)\s+(<.*?>)\s*$')
# file: line
fl = re.compile(r'^([^:]+):(\d+)\s*$')
# at file: line
afl = re.compile(r'^\s*at\s+([^\s:]+):(\d+)\s*$')


def load_from_strings(line, extras=None):
    frame = {}
    matched = False
    try:
        if not matched:
            match = naifafl.match(line)
            if match is not None:
                frame['depth'] = int(match.group(1))
                frame['address'] = match.group(2)
                frame['function'] = match.group(3)
                frame['args'] = match.group(4)
                frame['file'] = match.group(5)
                frame['dylib'] = match.group(6)
                matched = True
        if not matched:
            match = naifal.match(line)
            if match is not None:
                frame['depth'] = int(match.group(1))
                frame['address'] = match.group(2)
                frame['function'] = match.group(3)
                frame['args'] = match.group(4)
                frame['dylib'] = match.group(5)
                matched = True
        if not matched:
            match = nafal.match(line)
            if match is not None:
                frame['depth'] = int(match.group(1))
                frame['address'] = match.group(2)
                frame['function'] = match.group(3)
                frame['args'] = match.group(4)
                frame['dylib'] = match.group(5)
                matched = True
        if not matched:
            match = naifaf.match(line)
            if match is not None:
                frame['depth'] = int(match.group(1))
                frame['address'] = match.group(2)
                frame['function'] = match.group(3)
                frame['args'] = match.group(4)
                frame['file'] = match.group(5)
                matched = True
        if not matched:
            match = nfaf.match(line)
            if match is not None:
                frame['depth'] = int(match.group(1))
                frame['function'] = match.group(2)
                frame['args'] = match.group(3)
                frame['file'] = match.group(4)
                matched = True
        if not matched:
            match = naifa.match(line)
            if match is not None:
                assert ((not re.search(' at ', line))
                        or re.search('memory at ', line)
                        or re.search('at remote ', line)
                        ), line
                assert (not re.search(' from ', line))
                frame['depth'] = int(match.group(1))
                frame['address'] = match.group(2)
                frame['function'] = match.group(3)
                frame['args'] = match.group(4)
                matched = True
        if not matched:
            match = naif.match(line)
            if match is not None:
                assert (not re.search(' at ', line))
                assert (not re.search(' from ', line))
                assert (not re.search('\(.*?\)', line))
                frame['depth'] = int(match.group(1))
                frame['address'] = match.group(2)
                frame['function'] = match.group(3)
                matched = True
        if not matched:
            match = nfa.match(line)
            if match is not None:
                assert ((not re.search(' at ', line))
                        or re.search('memory at ', line)
                        or re.search('at remote ', line)
                        ), line
                assert (not re.search(' from ', line))
                assert (not re.search(' ()\s*$', line))
                frame['depth'] = int(match.group(1))
                frame['function'] = match.group(2)
                frame['args'] = match.group(3)
                matched = True
        if not matched:
            match = nf.match(line)
            if match is not None:
                assert (not re.search(' at ', line))
                assert (not re.search(' from ', line))
                assert (not re.search('\(.*?\)', line))
                frame['depth'] = int(match.group(1))
                frame['function'] = match.group(2)
                matched = True
    except:
        logging.error(line)
        raise

    # if frame.get('function') == '??':
    #     frame['function'] = None

    leftover_extras = []
    if 'file' in frame:
        match = fl.match(frame['file'])
        if match is not None:
            frame['file'] = match.group(1)
            frame['fileline'] = match.group(2)
            # print(frame['file'] + " : " + frame['fileline'], file=sys.stderr)
    elif extras is not None:
        for extra in extras:
            extra_matched = False
            if not extra_matched:
                match = afl.match(extra)
                if match is not None:
                    frame['file'] = match.group(1)
                    frame['fileline'] = match.group(2)
                    extra_matched = True
            if not extra_matched:
                leftover_extras.append(extra)

    if len(leftover_extras) > 0:
        frame['extra'] = leftover_extras

    if matched:
        return frame
    else:
        raise RuntimeError("Couldn't recognize stack frame format: %s" % (line.encode('unicode_escape')))


def load_from_file(path):
    encoding_guess = 'ISO-8859-1'
    if 'gz' in path:
        # gzip doesn't support encoding= ... this may need a workaround
        with gzip.open(path) as stackfile:
            stacklines = [fixline(line) for line in stackfile.readlines()]
    else:
        with codecs.open(path, encoding=encoding_guess) as stackfile:
            stacklines = stackfile.readlines()
    extras = []
    prevline = None
    stack = []

    for line in stacklines:
        line = line.rstrip()
        # for ch in line.lstrip():
        # if ch != '\t' and unicodedata.category(ch)[0] == 'C':
        # raise ValueError("Bad encoding %s %s: %s" % (encoding_guess, ch.encode('unicode_escape'), line.encode('unicode_escape')))
        if re.match('^#', line):
            if prevline is not None:
                stack.append(load_from_strings(prevline, extras))
            prevline = line
            extras = []
        if re.match('rax', line):
            return None
        else:
            extras.append(line.rstrip())

    if prevline is not None:
        stack.append(load_from_strings(prevline, extras))

    return stack


def parse_stacktrace_file(text, filepath=""):
    """
    Examples of function calls found in Stacktrace.txt.1:
        #12 0xb78a9b5d in IA__g_object_notify (object=0x9133590,
        #13 0xb7b62eb8 in IA__gdk_display_manager_set_default_display (
        #14 0xb7b60bbc in IA__gdk_display_open_default_libgtk_only ()
        #0  __GI___libc_free (mem=0x3) at malloc.c:2892
        ar_ptr = <optimized out>
        p = <optimized out>
        #0  0x00000000 in ?? ()
        #2  0x0601ffef in nux::GpuDevice::CreateAsmVertexShader (this=0xa034798) at ./GpuDeviceShader.cpp:47 ptr = (class nux::IOpenGLAsmVertexShader *) 0xa035b28 h = {ptr_ = 0x41b3217, _reference_count = 0xbfa54dbc,  _weak_reference_count = 0x603f3a4, _objectptr_count = 0xa033df0,  _destroyed = 0x0}
        #3  0x0601bda7 in IOpenGLAsmShaderProgram (this=0xa035b28, ShaderProgramName= {m_string = {static npos = <optimized out>, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfa54dbc "\004X\003\n�\03$
        No locals.
        #4  0x01c35211 in Gfx::opBeginImage(Object*, int) () from /usr/lib/libpoppler.so.12
        No symbol table info available.
        #5  0x01c2aae6 in Gfx::execOp(Object*, Object*, int) () from /usr/lib/libpoppler.so.12
        #0  0x00007f695f2367fc in QMutex::lock (this=0xc0a090)
        #1  0x00007f69505d48b2 in Soprano::Virtuoso::QueryResultIteratorBackend::close
        #2  0x00007f695b4e7226 in ~Iterator (this=0x7f694800acf0)
        #2  0x080c771b in xf86SigHandler ()
        #3  <signal handler called>
        #7  <signusername handler cusernameled>
        #1  0x00002b88b1a23599 in lucene::util::Compare::Char::operator() () from /usr/lib/libclucene.so.0
        #2  0x00002b88b1a358ee in std::_Rb_tree<char const*, std::pair<char const* const, void*>, std::_Select1st<std::pair<char const* const, void*> >, lucene::util::Compare::Char, std::allocator<std::pair<char const* const, void*> > >::insert_unique
        #3  0x00002b88b1a340c7 in lucene::store::TransactionalRAMDirectory::createOutput () from /usr/lib/libclucene.so.0
        #19 0x00002b625d2bfb31 in gnash::NetConnection::openConnection (this=<value optimized out>, url=<value optimized out>) at NetConnection.cpp:103 newurl = {static npos = 18446744073709551615, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>},  _M_p = 0x1000 <Address 0x1000
        #20 0x00002b625d2eb966 in gnash::NetStreamGst::startPlayback (this=0xd986b0) at NetStreamGst.cpp:910 head = "\b\021" video = <value optimized out> sound = <value optimized out> __PRETTY_FUNCTION__ = "void gnash::NetStreamGst::startPlayback()"
    """
    r = re.compile(
        r"^#([0-9]+) +(([a-zA-Z0-9]+) +in +)?((.+?) *[(].*?|(<sig[\w]+? handler [\w ]+?>)|([\w:<*,> \[\]-]+))(( +(at|from) +(\S+))|$)",
        re.MULTILINE)
    text = text.replace('\t', ' ')

    # remove new line from the file except when a new function call is declared.
    text = re.sub("\n +", " ", text)

    call_numbers = []
    fc = []

    for m in r.finditer(text):
        call_number = int(m.group(1))

        if call_number == 0 and len(call_numbers) > 0:
            # There are some files that contain two duplicate (Some contains stack traces from threads). Pick the first one.
            logging.getLogger().warning("{} contains more than one stack trace.".format(filepath))
            break

        call_numbers.append(call_number)

        method = None

        for i in range(5, 8):
            if m.group(i) is not None:
                method = m.group(i)
                break

        if method is None:
            raise Exception("One frame is None. {}\t{}.".format(m.group(), filepath))

        fc.append({
            # "mem_address": m.group(3),
            "function": method.strip(),
            # "params": m.group(4),
            "file": m.group(11),
        })

    for idx, cn in enumerate(call_numbers):
        if cn != idx:
            logging.getLogger().warning("Stack Trace is incomplete. {}".format(text))
            last = 0
            # There are some stack traces that are missing some calls e we complete it with None
            for idx, cn in enumerate(call_numbers):
                diff = cn - last
                if diff != 0:
                    i = idx
                    for _ in range(diff):
                        # fc.insert(i, {"mem_address": None, "method": None, "source": None, })
                        fc.insert(i, {"function": None, "file": None, })
                        i += 1
                last = cn + 1
            break

    return fc


def parse_stacktrace_top(text):
    re_stack_top = re.compile(r"(.+?) *[(].*?(( +(at|from) +(\S+))|$)", re.MULTILINE)
    fc = []
    for m in re_stack_top.finditer(text):
        fc.append({
            # "mem_address": None,
            "function": m.group(1).strip(),
            # "params": m.group(4),
            "file": m.group(5),
        })

    return fc


def parse_thread_stacktrace(text, file_path=""):
    re_thread = re.compile(r"^Thread +[0-9]+", flags=re.IGNORECASE | re.MULTILINE)
    threads = []

    matches = list(re_thread.finditer(text))

    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        threads.append(parse_stacktrace_file(text[start:end], file_path))

    return threads


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', help="Folder that contains all bug reports")
    parser.add_argument('output', help="Json File")
    parser.add_argument('date_file', help="Json File with date")
    parser.add_argument('lp_json', help="Data used by Campbell")

    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    args = parser.parse_args()
    logger.info(args)

    problem_re = re.compile(r"ProblemType: +[\w:. ]+\n")
    field_re = re.compile(r"^[.\w]+:", flags=re.MULTILINE)
    categorical_fields = set()
    reports = []
    oracle = {}

    n_before_crashes = 0
    n_crashes_buckets = 0
    n_buckets_more_one = 0

    for key, value in json.load(open(args.lp_json))['oracle'].items():
        crash_id = int(key.split(':')[1])
        oracle[crash_id] = value['bucket']
        # bpl.add(value['bucket'])

    with open(args.date_file) as f:
        date_by_bug = {}

        for bug_id, date in json.load(f):
            date_by_bug[bug_id] = date

    # Walking through the buckets
    for bucket_dir in os.listdir(args.folder):
        bucket_dir_path = os.path.join(args.folder, bucket_dir)
        bucket = []

        for report_dir in os.listdir(bucket_dir_path):
            n_before_crashes += 1

            # Read crash report data
            report_dir_path = os.path.join(bucket_dir_path, report_dir)
            report_id = int(report_dir)
            crash_report = {'bug_id': int(report_dir)}

            f = codecs.open(os.path.join(report_dir_path, 'Post.txt'), encoding="ISO-8859-1")
            content = "".join(f.readlines())

            # Extract content from Post.txt

            # Extract description
            mb = problem_re.search(content)

            # Reports like 362127 just only contain description.
            b = len(content) if mb is None else mb.start()

            crash_report['description'] = content[:b]

            # Extract categorical information
            categorical_info = content[b:]
            all_matches = [m for m in field_re.finditer(categorical_info)]
            l = len(all_matches)

            for i, m in enumerate(all_matches):
                start = m.end()
                last = all_matches[i + 1].start() if i + 1 < l else len(categorical_info)
                field_name = m.group()[:-1]

                categorical_fields.add(field_name)
                crash_report[field_name.lower()] = categorical_info[start:last].strip()

            creation_date = date_by_bug.get(crash_report['bug_id'])

            if creation_date is None:
                logger.warning("The creation date is missing for crash {}".format(crash_report['bug_id']))
                creation_date = datetime.strptime(crash_report['date'] + " +0000", '%a %b %d %H:%M:%S %Y %z')
            else:
                creation_date = datetime.strptime(creation_date, '%Y-%m-%d %H:%M:%S %z')

            if crash_report.get('title') is None:
                crash_report['short_desc'] = ''
            else:
                crash_report['short_desc'] = crash_report['title']
                del crash_report['title']

            crash_report["creation_ts"] = creation_date.timestamp()

            # Same extraction step of Party crash
            parsed_files = {}

            try_files = [
                "Stacktrace.txt (retraced)",
                "StacktraceSource.txt",
                "Stacktrace.txt",
                "Stacktrace",
                "Stacktrace.gz",
                "Stacktrace.txt.1",
            ]

            first_crash = None

            for trace_file in try_files:
                stack_path = os.path.join(report_dir_path, trace_file)
                if os.path.isfile(stack_path):
                    try:
                        trace = load_from_file(stack_path)
                    except Exception as e:
                        if first_crash is None:
                            first_crash = e
                        continue
                    if trace is None:
                        logger.error(stack_path + " did not contain a stack trace!")
                        if first_crash is not None:
                            raise first_crash
                    else:
                        break

            if trace is None:
                continue
                logger.warning("No stacktrace file found in %s. Ignore it." % (report_dir_path))
            else:
                crash_report['stacktrace'] = [trace]

            # Match: 'ThreadStacktrace.txt', "ThreadStacktrace.txt (retraced)", "ThreadStacktrace.txt.1", 'ThreadStacktrace.txt (retraced).1', Sanitized "ThreadStacktrace.txt (retraced)"
            match_threadstack = re.compile(r'(sanitized \")?ThreadStacktrace\.txt( *\(retraced\))?(\.[0-9]+)?(\")?',
                                           re.IGNORECASE)
            crash_report['threads'] = []

            for fname in os.listdir(report_dir_path):
                f = codecs.open(os.path.join(report_dir_path, fname), encoding="ISO-8859-1")
                content = "".join(f.readlines())

                if match_threadstack.match(fname) or fname == 'ThreadStacktrace(retraced).txt':
                    # Stack trace of each thread
                    crash_report['threads'] = parse_thread_stacktrace(content, os.path.join(report_dir_path, fname))

            # Add sorted stack traces
            crash_report['stacktrace'] = [{"exception": None, "message": None, "file": stack_path, "frames": trace}]

            # Add to the repository
            bucket.append((crash_report['creation_ts'], crash_report))

        # Remove crash reports that were not manually labeled
        if len(bucket) > 1:
            old_bucket = bucket
            bucket = []

            for t in old_bucket:
                if t[1]['bug_id'] not in oracle:
                    logging.getLogger().info(
                        "{} was removed since it was manually labeled (bucket: {})".format(t[1]['bug_id'], bucket_dir))
                    continue

                bucket.append(t)

            if len(bucket) != 0:
                n_crashes_buckets += len(bucket)
                n_buckets_more_one += 1

        # None of the reports in the bucket are in the oracle
        if len(bucket) == 0:
            logging.getLogger().warning(
                "All reporst from bucket {} were removed".format(bucket_dir))
            continue

        # Master report is the oldest report and the others ones are the duplicate
        oldest_report = min(bucket, key=lambda k: k[0])[1]

        for _, cr in bucket:
            if cr != oldest_report:
                cr['dup_id'] = oldest_report['bug_id']
            else:
                cr['dup_id'] = None

            reports.append(cr)

    logger.info("Number of crashes in the original dataset: {} ".format(n_before_crashes))
    logger.info("Number of crashes after preprocessing: {} ".format(len(reports)))
    logger.info("Number of buckets with more than one crash: {} ".format(n_buckets_more_one))
    logger.info("Number of duplicate crash (considering the master) :{} ".format(n_crashes_buckets))


    with codecs.open(args.output, 'w') as f:
        json.dump(reports, f)
