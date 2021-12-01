#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters are classes that change the token, like transform to lower case the letters 
"""
import hashlib
import importlib
import json
import logging
import re
import string
from collections import defaultdict
from math import log

import numpy as np
from gensim.matutils import unitvec
from nltk import WhitespaceTokenizer, word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

# REGEX

CLASS_DEF_JAVA_REGEX = r'((public|private)\s*)?class\s+(\w+)(\s+extends\s+(\w+))?(\s+implements\s+([\w,\s]+))?\s+\{(\s*})?'
FUNC_IF_DEF_JAVA_REGEX = r'\w+\s*\([^)]*\)\s*\{'
IF_JAVA_REGEX = r'if\s*\(.*\{'
OBJ_JAVA_REGEX = r'\w\s*=\snew[^;]+;'
# SPLIT_PUNCT_REGEX = r'(\w)([!#$%&()*+,-./:;<=>?@[\]^_`{|}~])(\S)'
SPLIT_PUNCT_REGEX = r"([\"!#\\$%&()*+,-./:;<=>?@[\]^_'`{|}~])"
RVM_REPEATED_PUNC = "([%s])\\1{1,}" % string.punctuation


def tokenize_field(tokenizer, field_value):
    if field_value is None:
        field_value = NONE_TOKEN

    return tokenizer(field_value)


def checkDesc(desc):
    return desc.strip() if desc and len(desc) > 0 else ""


# Filters
class Filter:

    def filter(self, token, sentence):
        raise NotImplementedError()

    def getSymbol(self):
        return None


class TransformLowerCaseFilter(Filter):
    def filter(self, token, sentence):
        return token.lower()


class TransformNumberToZeroFilter(Filter):
    def filter(self, token, sentence):
        return re.sub('[0-9]', '0', token)


class ReduceNumberToZeroFilter(Filter):
    def filter(self, token, sentence):
        return re.sub('[0-9]+', '0', token)


class MultiLineTokenizer(object):
    """Tokenize multi line texts. It removes the new lines and insert all words in single list """

    def __call__(self, text):
        return word_tokenize(text)

    def __call__(self, text):
        return word_tokenize(text)


class WhitespaceTokenizer(object):
    """Tokenize multi line texts. It removes the new lines and insert all words in single list """

    def __init__(self):
        self.regexp = re.compile(r'\s+')

    def __call__(self, text):
        return [tok for tok in self.regexp.split(text) if tok]


class CamelCaseWithPuncTokenizer(object):
    """
    Separate the punctuations and non-punctuations by space and break the function name that use the camel case caps.
    ex: PluginClassHandler<GridWindow, CompWindow, 0>::get =>
	    Plugin, Class, Handler, <, Grid, Window, ',', Comp, Window,',', 0, >, ::, get
    """

    def __call__(self, text):
        if len(text) == 0:
            return []
        tokens = []
        token_chars = []
        types = [-1, -1, -1]  # previous_previous, previous and current

        for char in text:
            if char.isspace():
                # Char is a whitespace
                if len(token_chars) == 0:
                    # Remove space in the begin of text
                    continue
                types[2] = 0
            elif char.isnumeric():
                # Char is a number
                types[2] = 1
            elif char.isalpha():
                if char.isupper():
                    # Char is a upper case alphabet
                    types[2] = 2
                else:
                    # Char is a lower case alphabet
                    types[2] = 3
            else:
                # Char is a symbol
                types[2] = 4

            if len(token_chars) > 0:
                if types[2] == 4 and types[1] == 4 and previos_char != char:
                    # Separate different symbols (ex: $$ => $ or $, => $ ,)
                    tokens.append(''.join(token_chars))
                    token_chars = []
                elif types[1] != types[2]:
                    # Previous and current characteres are not the same type. Should we separate token from current char?
                    if types[1] == 0:
                        # Since previous char was a space, current is a space too. Ignore them!
                        token_chars = []
                    elif types[1] != 2 or types[2] != 3:
                        """
                        Separate if the current and previous are different except when 
                            previous is upper and current is lower.
                        ex: 123a => 123 a; CamelC => Camel C;
                        """
                        tokens.append(''.join(token_chars))
                        token_chars = []
                    elif types[0] == 2:
                        """
                        Previous is upper and current is lower. However the antepenultimate is upper too. 
                        Separete previous char from the rest.
                        ex: IPQu => IP Qu where current=u, previous=Q and ante
                        """
                        previos_char = token_chars.pop()
                        tokens.append(''.join(token_chars))
                        token_chars = [previos_char]

            # Update types
            types[0] = types[1]
            types[1] = types[2]

            # Add token
            token_chars.append(char)

            # Previous char
            previos_char = char

        # Leftover. Ignore if they are space:
        if types[1] != 0:
            tokens.append(''.join(token_chars))

        return tokens


class NonAlphaNumCharTokenizer(object):
    """
    Replace the non alpha numeric character by space and tokenize the sentence by space.
    For example: the sentence 'hello world, org.eclipse.core.launcher.main.main' is tokenized to
    [hello, word , org, eclipse, core, launcher, main, main ].
    """
    REGEX = re.compile('[\W_]+', re.UNICODE)

    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()

    def __call__(self, text):
        text = re.sub(NonAlphaNumericalChar.REGEX, ' ', text)
        return self.tokenizer(text)


class HTMLSymbolFilter(Filter):

    def filter(self, token, sentence):
        return re.sub(r"((&quot)|(&gt)|(&lt)|(&amp)|(&nbsp)|(&copy)|(&reg))+", '', token)


class StopWordRemoval(Filter):

    def __init__(self, stopwords=stopwords.words('english')):
        self.stopwords = stopwords

    def filter(self, token, sentence):
        if token in self.stopWords:
            return ''

        return token


class NeutralQuotesFilter(Filter):
    """
    Transform neutral quotes("'`) to opening(``) or closing quotes('')
    """

    def __init__(self):
        super(NeutralQuotesFilter, self).__init__()
        self.__lastSentence = ""
        self.__isNextQuoteOpen = True

    def filter(self, token, sentence):
        if re.search(r"^[\"`']$", token):
            if self.__lastSentence == sentence:
                if self.__isNextQuoteOpen:
                    self.__isNextQuoteOpen = False
                    return "``"
                else:
                    self.__isNextQuoteOpen = True
                    return "''"
            else:
                self.__lastSentence = sentence
                self.__isNextQuoteOpen = False
                return "``"

        return token


class ModuleFunctionFilter(Filter):
    SYMBOL = '#MODULE_FUNCTION'

    def filter(self, token, sentence):
        a = str(token)
        # todo: this filter is comparison with files (a.java, p.php, s.jsp, a.txt)
        pattern = r'^[a-zA-Z][a-zA-Z0-9]+(\.[a-zA-Z][a-zA-Z0-9]+)+(\(\))?$'
        token = re.sub(pattern, ModuleFunctionFilter.SYMBOL, token)

        return token

    def getSymbol(self):
        return self.SYMBOL


class NonAlphaNumericalChar(Filter):
    "Remove all non alpha numerical character"
    REGEX = re.compile('[\W_]+', re.UNICODE)

    def __init__(self, repl=' '):
        self.repl = repl

    def filter(self, token, sentence):
        token = re.sub(NonAlphaNumericalChar.REGEX, self.repl, token)

        return token


class UrlFilter(Filter):
    """
    Tokens starting with “www.”, “http.” or ending with “.org”, “.com” e ".net" are converted to a “#URL” symbol
    """

    SYMBOL = "#URL"

    def filter(self, token, sentence):
        a = str(token)
        token = re.sub(r"^((https?:\/\/)|(www\.))[^\s]+$", "#URL", token)
        # token = re.sub(r"^[^\s]+(\.com|\.net|\.org)\b([-a-zA-Z0-9@;:%_\+.~#?&//=]*)$", "#URL", token)
        token = re.sub(r"^[^\s]+(\.com|\.net|\.org)([/?]([-a-zA-Z0-9@;:%_\+.~#?&//=]*))?$", "#URL", token)

        return token

    def getSymbol(self):
        return self.SYMBOL


class RepeatedPunctuationFilter(Filter):
    """
    Repeated punctuations such as “!!!!” are collapsed into one.
    """

    def filter(self, token, sentence):
        token = re.sub(r"^([,:;><!?=_\\\/])\1{1,}$", '\\1', token)
        token = re.sub(r"^[.]{4,}$", "...", token)
        token = re.sub(r"^[.]{2,2}$", ".", token)
        token = re.sub(r"^[--]{3,}$", "--", token)

        return token


class RemovePunctuationFilter(Filter):
    REGEX = re.compile(SPLIT_PUNCT_REGEX)

    def filter(self, token, sentence):
        token = self.REGEX.sub('', token)

        return token


class StripPunctuactionFilter(Filter):
    REGEX = r"""(^[]\d!"#$%&'()*+,-.\/:;<=>?@[\^_`{|}~]+)|([]\d!"#$%&'()*+,-.\/:;<=>?@[\^_`{|}~]+$)"""

    def filter(self, token, sentence):
        token = re.sub(self.REGEX, '', token)

        return token


class DectectNotUsualWordFilter(Filter):
    puncSet = set(string.punctuation)

    def filter(self, token, sentence):
        # Remove sentences which 20% of characters are numbers or punctuations
        npt = 0

        if len(token) == 0:
            return token

        for c in token:
            if c.isnumeric() or c in self.puncSet:
                npt += 1

        if float(npt) / len(token) > 0.20:
            return ''

        return token


def load_filters(filterNames):
    """
    Instance the filters using their names
    :param filterNames: class name
    :return:
    """
    filters = []
    module_ = importlib.import_module(Filter.__module__)

    for filterName in filterNames:
        filters.append(getattr(module_, filterName)())

    return filters


def cleanDescription(desc):
    # Remove class declaration
    cleanDesc = re.sub(CLASS_DEF_JAVA_REGEX, '', desc)

    # Remove if
    cleanDesc = re.sub(IF_JAVA_REGEX, '', cleanDesc)

    # Remove function, catch and some ifs
    cleanDesc = re.sub(FUNC_IF_DEF_JAVA_REGEX, '', cleanDesc)

    # Remove variablie
    cleanDesc = re.sub(OBJ_JAVA_REGEX, '', cleanDesc)

    # Remove time
    cleanDesc = re.sub(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}', '', cleanDesc)

    # Remove date
    cleanDesc = re.sub(r'[0-9]{1,4}([/-])[0-9]{1,2}\1[0-9]{2,4}', '', cleanDesc)

    # Remove repeated punctuation like ######
    cleanDesc = re.sub(r"([,:;><!?=_\\\/*-.,])\1{1,}", '\\1', cleanDesc)

    newdesc = ""
    puncSet = set(string.punctuation)

    for l in cleanDesc.split("\n"):
        # Remove sentence that have less 10 characters
        if len(l) < 10:
            continue

        # Remove the majority of stack traces, some code and too short texts. Remove sentence that has less 5 tokens.
        nTok = 0
        for t in re.split(r'\s', l):
            if len(t) > 0:
                nTok += 1

        if nTok < 5:
            continue

        # Remove sentences which 20% of characters are numbers or punctuations
        npt = 0
        for c in l:
            if c.isnumeric() or c in puncSet:
                npt += 1

        if float(npt) / len(l) > 0.20:
            continue

        newdesc += l + '\n'

    return newdesc


def softClean(text, rmPunc=False, sentTok=False, rmNumber=False):
    cleanText = re.sub(RVM_REPEATED_PUNC, '\\1', text)

    # Remove time
    cleanText = re.sub(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}', '', cleanText)
    # Remove date
    cleanText = re.sub(r'[0-9]{1,4}([/-])[0-9]{1,2}\1[0-9]{2,4}', '', cleanText)

    if rmNumber:
        cleanText = re.sub(r'[0-9]', '', cleanText)

    if sentTok:
        cleanText = '\n'.join(sent_tokenize(text))

    if rmPunc:
        cleanDescSpaced = re.sub(SPLIT_PUNCT_REGEX, ' ', cleanText)
    else:
        cleanDescSpaced = re.sub(SPLIT_PUNCT_REGEX, ' \\1 ', cleanText)

    return cleanDescSpaced


NONE_TOKEN = "NNNOOONNNEEE"

"""
Preprocess stacktraces considering function as unit
"""


def preprocess_function(func_name):
    # last_func = func_name
    # Strip arguments from function_name
    if isinstance(func_name, int):
        return func_name

    func_name = re.sub(r'\(.*\)', '', func_name)
    # Strip __GI__ in the begining of function call
    # todo:  It doesn't clean '*__GI_raise') '*__GI_abort')
    func_name = re.sub(r'^_*GI_+', '', func_name)
    # Strip _ in the begining of function call
    func_name = re.sub(r'^_+', '', func_name)
    # todo: remove ^IA__

    # if last_func != func_name:
    #     print("{} => {}".format(last_func, func_name))

    return func_name


def fn_preprocess_package(vocab, frames, ukn_tkn, ukn_id, max_depth):
    function_ids = []

    for depth, frame in enumerate(frames):
        func_name = frame.get('function')

        if func_name is None:
            continue

        if depth >= max_depth:
            break

        if not isinstance(func_name, int):
            parts = func_name.split(".")

            if len(parts) < 3:
                continue

            package_name = ".".join(parts[:-2])
        else:
            package_name = func_name

        token_id = vocab.setdefault(preprocess_function(package_name), len(vocab))

        function_ids.append(token_id)

    if len(function_ids) == 0:
        function_ids.append(vocab.setdefault(f"@@@{len(vocab)}@@@", len(vocab)))

    return function_ids


def std_function_preprocess(vocab, frames, ukn_tkn, ukn_id, max_depth):
    function_ids = []

    for depth, frame in enumerate(frames):
        func_name = frame.get('function')

        if depth >= max_depth:
            break

        if func_name is None or func_name == ukn_tkn:
            token_id = ukn_id
        elif func_name is not None:
            token_id = vocab.setdefault(preprocess_function(func_name), len(vocab))

        function_ids.append(token_id)

    return function_ids


class BartzPreprocessor(object):

    def __init__(self, module_source):
        self.module_source = module_source

    def preprocess(self, vocab, frames, ukn_tkn, ukn_id, max_depth):
        preprocessed_frames = []

        for depth, frame in enumerate(frames):
            if depth >= max_depth:
                break

            func_name = frame.get('function')

            if self.module_source == "package":
                # Java package
                parts = func_name.split('.')
                module_name = ".".join(parts[:-2]) if len(parts) > 2 else None
                func_name = ".".join(parts[-2:])
            elif self.module_source == "class":
                # Java class and package
                parts = func_name.split('.')
                module_name = ".".join(parts[:-1]) if len(parts) > 1 else None
                func_name = parts[-1]
            elif self.module_source == "namespace":
                # Is difficult to differentiate namespace and class names. We'll consider the same
                # unity::dash::SearchBar::~SearchBar()
                parts = func_name.split('::')
                module_name = "::".join(parts[:-1]) if len(parts) > 1 else None
                func_name = parts[-1]
            elif self.module_source == "file":
                # Filename and dynamic library
                if frame.get('file') is not None:
                    module_name = frame['file'].split("/")[-1]
                elif frame.get('dylib') is not None:
                    module_name = frame['dylib'].split("/")[-1]
                else:
                    module_name = None

            if module_name is None or module_name == ukn_tkn:
                module_id = ukn_id
            else:
                module_id = vocab.setdefault(module_name, len(vocab))

            if func_name is None or func_name == ukn_tkn:
                function_id = ukn_id
            elif func_name is not None:
                func_name = preprocess_function(func_name)
                function_id = vocab.setdefault(preprocess_function(func_name), len(vocab))

            preprocessed_frames.append([function_id, module_id])

        return np.asarray(preprocessed_frames, dtype=np.uint32)


def rm_duplicate_stacks(frames, st_set=None):
    if st_set == None:
        st_set = set()

    r = []

    for st in frames:
        st_hash = hashlib.sha512(json.dumps(st).encode("utf-8")).hexdigest()

        if st_hash in st_set:
            continue

        st_set.add(st_hash)
        r.append(st)

    return r


def check_stacktraces(stacktrace, report_id, is_nested):
    """
    Check whether stacktrace from different threads are concatenated or it is missing frames in stacktrace

    :param stacktraces:
    :param max_depth:
    :param report_id:
    :param is_nested:
    :return:
    """
    """
    Check whether stacktrace from different threads are concatenated or it is missing frames in stacktrace

    :param stacktraces:
    :param max_depth:
    :param report_id:
    :param is_nested:
    :return:
    """
    new_stacktrace = []
    last_depth = -1

    # Split stacktraces:
    for frame in stacktrace['frames']:
        depth = int(frame['depth'])
        diff = depth - last_depth - 1

        if diff > 0:
            # Fix stack trace
            new_stacktrace.extend(({'depth': last_depth + i + 1} for i in range(diff)))
            logging.getLogger().warning("report stacktrace {} is missing frames.".format(report_id))
        elif diff < 0:
            if len(new_stacktrace) > 0:
                yield new_stacktrace

            last_depth = depth
            new_stacktrace = [{'depth': i} for i in range(depth)]
            # logging.getLogger().error("More than one stack in report {}.".format(report_id, stacktrace['frames']))

        last_depth = depth
        new_stacktrace.append(frame)

    if (new_stacktrace is None or len(new_stacktrace) == 0) and not is_nested:
        logging.getLogger().warning("preprocessed_frame is empty for report {}".format(report_id))
        return

    yield new_stacktrace


def preprocess_stacktrace(all_report_ids, report_db, max_depth, filter_recursion=None, return_vocab=False, ukn_tkn="??",
                          vocab=None, stacktraces_by_id=None, unique_ukn_report=True,
                          preprocess_func=std_function_preprocess, merge_main_nested=False, ukn_set=None,
                          rm_dup_stacks=False):
    # Convert the
    vocab = {} if vocab is None else vocab
    stacktraces_by_id = {} if stacktraces_by_id is None else stacktraces_by_id
    ukn_tkn_main = ukn_tkn

    for report_id in all_report_ids:
        stacktraces = report_db.get_report(report_id)['stacktrace']

        if unique_ukn_report:
            # Create token for unknown function name. Give a unique id to the unknown values in stacktrace
            # Following ABRT, we only compare the function names and consider two unknown function(??) as different
            ukn_tkn = '{}{}'.format(ukn_tkn_main, report_id)

            if ukn_tkn in vocab:
                raise Exception("Token for unknown function name in the report {} already exists.".format(report_id))

        ukn_id = vocab.setdefault(ukn_tkn, len(vocab))

        if ukn_set is not None:
            ukn_set.add(ukn_id)

        if not isinstance(stacktraces, list):
            stacktraces = [stacktraces]

        main_frames = []
        nested_frames = []

        for stacktrace in stacktraces:
            main_frames.extend(check_stacktraces(stacktrace, report_id, False))

            if len(stacktrace['nested']) > 0:
                nested_frames.extend(check_stacktraces(stacktrace['nested'][0], report_id, True))

        for i in range(len(main_frames)):
            preprocessed_frame = preprocess_func(vocab, main_frames[i], ukn_tkn, ukn_id, max_depth)
            main_frames[i] = remove_recursive_calls(preprocessed_frame, ukn_id, filter_recursion)

        for i in range(len(nested_frames)):
            preprocessed_frame = preprocess_func(vocab, nested_frames[i], ukn_tkn, ukn_id, max_depth)
            nested_frames[i] = remove_recursive_calls(preprocessed_frame, ukn_id, filter_recursion)

        if len(main_frames) == 0 and len(nested_frames) == 0:
            raise Exception("{} contains 0 zero stacktraces".format(report_id))
        else:
            sum_lengths = 0

            for fr in main_frames:
                sum_lengths += len(fr)

            for fr in nested_frames:
                sum_lengths += len(fr)

            if sum_lengths == 0:
                raise Exception("{} contains 0 zero stacktraces".format(report_id))

        if rm_dup_stacks:
            st_set = set() if merge_main_nested else None

            main_frames = rm_duplicate_stacks(main_frames, st_set)
            nested_frames = rm_duplicate_stacks(nested_frames, st_set)

        stacktraces_by_id[report_id] = main_frames + nested_frames if merge_main_nested else (
            main_frames, nested_frames)

    if return_vocab:
        return stacktraces_by_id, vocab

    # return stacktraces_by_id, vocab
    return stacktraces_by_id


def filter_by_document_frequency(stacktraces_by_id, df_array, threshold):
    if threshold < 1:
        return

    ignore_function = set((f_id for f_id, df in enumerate(df_array) if df > threshold))

    if len(ignore_function) == 0:
        return

    for stack_id, (preprocessed_frames, nested_frames) in stacktraces_by_id.items():
        for idx, frame in enumerate(preprocessed_frames):
            new = [w for w in frame if w not in ignore_function]

            if len(new) != len(frame):
                preprocessed_frames[idx] = new

        for idx, frame in enumerate(nested_frames):
            new = [w for w in frame if w not in ignore_function]

            if len(new) != len(frame):
                nested_frames[idx] = new


def remove_recursive_calls(frames: [int] or [{}] or [[]], ukn_function, recursion_removal):
    # todo: we need to fix it
    if recursion_removal is None:
        return frames

    is_int = isinstance(frames[0], int)

    if recursion_removal == "brodie":
        # Quickly Finding Known Software Problems via Automated Symptom Matching - Mark Brodie 2006
        previous_function = None
        clean_stack = []

        for idx, fr in enumerate(frames):
            func_name = fr if is_int else fr['function']

            if func_name == ukn_function or func_name != previous_function:
                clean_stack.append(fr)

            previous_function = func_name
    elif recursion_removal == "modani":
        # Automatically Identifying Known Software Problems - Natwar Modani 2007
        idx = 0
        clean_stack = []

        while idx < len(frames):
            fr = frames[idx]
            func_name = fr if is_int else fr['function']
            end_idx = None

            clean_stack.append(fr)

            if func_name != ukn_function:
                for bk_idx in range(len(frames) - 1, idx, -1):
                    bk_frame = frames[bk_idx]
                    bk_func = bk_frame if is_int else bk_frame['function']

                    if func_name == bk_func:
                        end_idx = bk_idx
                        break

            if end_idx is not None:
                idx = end_idx + 1
            else:
                idx += 1
    else:
        raise Exception("Invalid argument value for recursion_removal: {}".format(recursion_removal))

    frames = clean_stack
    return frames


def retrieve_df(stacktraces, to_np=True, idf_list=None, extra=0, freq_by_stacks=False, n_docs=0):
    df = defaultdict(int)
    largest_id = -1

    for stacks in stacktraces:
        if isinstance(stacks, tuple):
            main_stack, nested = stacks
        else:
            main_stack = stacks
            nested = []

        if freq_by_stacks:
            for stacktrace in main_stack + nested:
                for function in set(stacktrace):
                    if function > largest_id:
                        largest_id = function

                    df[function] += 1

                n_docs += 1
        else:
            function_set = set()
            n_docs += 1

            for frames in main_stack:
                function_set.update(frames)

            for frames in nested:
                function_set.update(frames)

            for function in function_set:
                if function > largest_id:
                    largest_id = function

                df[function] += 1

    n_tokens = largest_id + 1

    if idf_list is None:
        # Create a new one
        idf_list = np.zeros(n_tokens + extra, dtype=np.double) if to_np else [0.0] * (n_tokens)
    else:
        # Update Idf list
        if n_tokens > len(idf_list):
            if to_np:
                idf_aux = np.zeros(n_tokens + extra, dtype=np.double)

                for i in range(idf_list.shape[0]):
                    idf_aux[i] = idf_list[i]

                idf_list = idf_aux
            else:
                idf_list.extend((0.0 for _ in range(n_tokens - len(idf_list))))

    for k, v in df.items():
        idf_list[k] += v

    return idf_list, n_docs


def generate_doc_freq_matrix(field_name, db):
    tokenizer = NonAlphaNumCharTokenizer()
    vocab = {}
    rows = []
    num_nnz = 0
    df = list()

    for report in db:
        text = report.get(field_name, "")
        tkns = tokenizer(text.lower())
        counter = defaultdict(int)

        for tkn in tkns:
            tkn_id = vocab.setdefault(tkn, len(vocab))
            counter[tkn_id] += 1

        bow = sorted(counter.items())
        num_nnz += len(bow)
        for tkn_id, freq in bow:
            if tkn_id >= len(df):
                df.append(0)

            df[tkn_id] += 1
        rows.append(bow)

    if num_nnz == 0:
        return None, None

    n = len(rows)

    for i in range(len(df)):
        df[i] = log(n / df[i])

    for i in range(len(rows)):
        for j in range(len(rows[i])):
            tkn_id = rows[i][j][0]
            rows[i][j] = (tkn_id, rows[i][j][1] * df[tkn_id])

        rows[i] = unitvec(rows[i])

    return (rows, len(vocab))
    # return corpus2csc((unitvec(v) for v in rows), num_terms=len(vocab), num_docs=len(rows),
    #                   num_nnz=num_nnz, dtype=np.float32).T

    # n = len(df)
    # for i in range(len(df)):
    #     df[i] /= n


def prepare_cmp_fields(report_db):
    fields = ['os', 'cpu', 'nonfreekernelmodules', 'packagearchitecture', 'proccmdline', 'proccwd',
              'sourcepackage', 'procenviron', 'package', 'signal', 'interpreterpath', 'usergroups', 'uname',
              'executablepath', 'disassembly', 'segvreason', 'segvanalysis', 'release', 'version',
              'procattrcurrent', 'os_version', 'machinetype', 'lsusb', 'livemediabuild',
              'relatedpackageversions', 'compizplugins', 'assertionmessage', 'checkboxsubmission',
              'checkboxsystem', 'xsessionerrors', 'installationmedia', 'dkmsstatus', 'prockernelcmdline',
              'distupgraded', 'distrovariant', 'graphicscard', 'distrocodename', 'compositorrunning',
              'upgradestatus', 'apportversion', 'installationdate', 'currentdesktop', 'product', 'component',
              'short_desc', 'priority', 'bug_severity']
    # fields = ['os', 'cpu', 'nonfreekernelmodules', 'packagearchitecture']
    global doc_freq_matrix_by_field
    doc_freq_matrix_by_field = []
    for f in fields:
        # start = time()
        doc_freq_matrix_by_field.append(generate_doc_freq_matrix(f, report_db))
        # print("Gene {}: {}".format(f, time() - start))
    global report2idx
    report2idx = dict(((report['bug_id'], idx) for idx, report in enumerate(iter(report_db))))


if __name__ == '__main__':
    print(remove_recursive_calls([1, 2, 3, 1, 2, 1, 2, 1, 2], -1, 2))
    print(remove_recursive_calls([1, 2, 1, 2, 1, 2, 3, 2, 3, 2, 3], -1, 2))
    print(remove_recursive_calls([4, 4, 4, 4, 4, 4, 5, 5, 5, 5], -1, 2))
    print(remove_recursive_calls([4, 4, 4, 4, 3, 2, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5], -1, 4))

    print(retrieve_df({0: [[[0, 2, 3, 3, 3, 4, 8]], [[4, 5, 10]]], 1: [[[0, 2, 2, 2, 4, 6, 10]], [[35, 35]]],
                       2: [[[2, 1, 2, 3]], [[4]]], 3: [[[0, 4, 4, 4, 4]], [[4, 5]]]}))
