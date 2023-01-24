import time
import traceback
import random
import logging
import itertools
import copy
import hashlib
import io
import os
import sys
import joblib
import inspect
import binascii
import asyncio
import ast
import tempfile
from collections import defaultdict, OrderedDict
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


try:
    import rich

    has_rich = True
except ImportError:
    has_rich = False

if has_rich:
    from rich.logging import RichHandler

    logger = logging.getLogger("mandala")
    logging_handler = RichHandler(enable_link_path=False)
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[logging_handler]
    )
else:
    logger = logging.getLogger("mandala")
    # logger.addHandler(logging.StreamHandler())
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)

sess = Session()

TableType = TypeVar("TableType", pa.Table, pd.DataFrame)


class InternalError(Exception):
    pass
