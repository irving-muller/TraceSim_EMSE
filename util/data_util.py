import importlib
import json
import logging
from datetime import datetime, timezone

import numpy as np

import data.preprocessing as preprocessing

def read_date_from_report(bug):
    return datetime.fromtimestamp(bug['creation_ts'], tz=timezone.utc)