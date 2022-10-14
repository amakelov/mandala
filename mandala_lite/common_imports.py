import traceback
import random
import logging
import itertools
import copy
import hashlib
import io
import os
import joblib
import inspect
import binascii
from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Callable,
    Tuple,
    Iterable,
    Optional,
    Set,
    Union,
    TypeVar,
)
from pathlib import Path

import pandas as pd
import pyarrow as pa
import numpy as np


class Session:
    # for debugging

    def __init__(self):
        self.items = []


logger = logging.getLogger("mandala_lite")
# logger.addHandler(logging.StreamHandler())
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)
sess = Session()

TableType = TypeVar("TableType", pa.Table, pd.DataFrame)


class SyncException(Exception):
    pass
