import argparse
import json
import logging
import re
from datetime import datetime


def search_extra(extra, regex):
    for e in extra:
        result = re.search(regex, e)

        if result is not None:
            yield result.group(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('original_file', help="Json path")
    parser.add_argument('new_file', help="Path of the new json")
    parser.add_argument('creation_date', help="file with the creation date of each report")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    json_obj = json.load(open(args.original_file))
    oracle = json_obj['oracle']
    crashes = json_obj['crashes']

    with open(args.creation_date) as f:
        date_by_bug = {}

        for bug_id, date in json.load(f):
            date_by_bug[bug_id] = int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S %z').timestamp())

    """
    report 754562 report was deleted in the launchpad. Since the report ids follows the submission order,
    we consider that creation date of the report 754562 is between 754533 (2011-04-08 11:34:43 UTC) 
    and 754567(2011-04-08 12:17:26 UTC).
    """
    date_by_bug[754562] = int(datetime.strptime('2011-04-08 12:00:00 +0000', '%Y-%m-%d %H:%M:%S %z').timestamp())

    for crash_id, crash in crashes.items():
        bug_id = re.search("[0-9]+", crash_id)

        if bug_id is None:
            raise Exception("Does {} have an ID?".format(bug_id))

        crash['bug_id'] = int(bug_id[0])
        crash['creation_ts'] = date_by_bug[crash['bug_id']]

    # Sort by creation date and id
    sorted_crashes = sorted(crashes.values(), key=lambda r: (r['creation_ts'], r['bug_id']))
    master_by_bucket = {}

    dataset = []

    for crash in sorted_crashes:
        bucket = oracle[crash['database_id']]['bucket']

        # remove crash without stacktraces
        if 'stacktrace' not in crash or len(crash['stacktrace']) == 0:
            print("{} was removed because it doesn't contain stacktrace.".format(crash['database_id']))
            continue

        # add crash to correct bucket
        if bucket not in master_by_bucket:
            master_by_bucket[bucket] = crash['bug_id']
            crash['dup_id'] = None
        else:
            master_id = master_by_bucket[bucket]
            crash['dup_id'] = master_id

        # lower case all fields
        fields = list(crash.items())

        for name, value in fields:
            new_name = name.lower()

            if name == new_name:
                continue

            del crash[name]

            if new_name == 'title':
                new_name = "short_desc"

            crash[new_name] = value

        # Correct frames. Some dynamic libraries or files are in the extras
        for frame in crash['stacktrace']:
            if len(frame.get('extra', [])) > 0 and frame.get('file') is None and frame.get('dylib') is None:
                # Extract dylib or file from extra
                extra = frame['extra']

                for dylib in search_extra(extra, "from +([^\s]+)"):
                    if dylib[-1] == '?':
                        #Remove ? from the end
                        dylib = dylib[:-1]

                    if re.search(r'\.(so|hr)(\.[0-9]+d?)*$', dylib) is None:
                        logger.warning("Consider dylib as false positive: dylib={}, depth={}, id={}".format(dylib, frame['depth'],
                                                                                                crash['database_id']))
                        continue

                    frame['dylib'] = dylib
                    break

                if frame.get('dylib',None) is None:
                    for file in search_extra(extra, "at +([^\s]+)"):
                        split = file.split(':')

                        if len(split) > 1:
                            file, fileline = split
                            fileline = int(re.match('([0-9]+)(\W+)?', fileline).group(1))
                            frame['fileline'] = fileline
                        else:
                            file = split[0]

                        if re.search("\.(cython_mod|tcc|cpp|h|hpp|m|moc|xs|xsi|S|cc)$|(ostream|iostream)$", file) is None:
                            logger.warning(
                                "Consider file as false positive: file={}, depth={}, id={}".format(file, frame['depth'],
                                                                                        crash['database_id']))
                            continue

                        frame['file'] = file
                        break

        # Add message, nested and exception to stacktrace
        stacktrace = {
            "message": None,
            "exception": None,
            "frames": crash['stacktrace'],
            "nested": []
        }

        crash['stacktrace'] = stacktrace
        dataset.append(crash)

    print("Nm of crashes: {}".format(len(dataset)))
    json.dump(dataset, open(args.new_file, 'w'))
