"""
It extracts stack traces in the description of bug reports.
We implemented the same regular expression in "Finding Duplicates of Your Yet Unwritten Bug Report. Johannes Lerch. 2013"
    that capture stack traces from JAVA.
"""

import argparse
import json
import logging


def fix_stacktrace(stacktrace):
    frames = stacktrace['function_calls']

    for depth, frame in enumerate(frames):
        frame['depth'] = depth
        file = frame['source']

        frame['function'] = frame['method']

        del frame['source']
        del frame['method']

        f = file.rsplit(':', 1)
        frame['file'] = f[0]

        if len(f) > 1:
            frame['fileline'] = f[1]

    if len(stacktrace.get('nested',[])) > 0:
        if len(stacktrace['nested']) > 1:
            print("There are {} nested exceptions".format(len(stacktrace['nested'])))

        stacktrace['nested'] = [fix_stacktrace(nested) for nested in stacktrace['nested']]

    stacktrace['frames'] = frames
    del stacktrace['function_calls']

    return stacktrace

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-input', required=True, help="Mongo database")
    parser.add_argument('-output', required=True, help="Mongo collection")

    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    args = parser.parse_args()
    logger.info(args)

    # Connect to mongo dataset
    input = json.load(open(args.input))

    for report in input:
        if isinstance(report['dup_id'], list):
            report['dup_id'] = None

        report["stacktrace"] = [fix_stacktrace(stacktrace) for stacktrace in report['stack_trace']]
        del report['stack_trace']


    json.dump(input, open(args.output, 'w'))