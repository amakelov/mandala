import typing
from .core.utils import CompatArg, AsType, AsTransient, AsDelayedStorage
from .core.bases import unwrap, detached
from .core.config import CoreConfig
from .core.wrap import wrap_detached
from .storages.kv_impl.sqlite_impl import SQLiteStorage
from .storages.kv_impl.joblib_impl import JoblibStorage
from .ui.storage import Storage
from .ui.execution import wrap
from .ui.context import (
    context, run, query, transient, delete, define, noop, retrace, capture
)
from .ui.funcs import op, superop
from .ui.vars import Var, Query, BuiltinVars
from .util.logging_ut import set_logging_level
from .queries.rel_weaver import ValQuery, ListQuery, DictQuery
from .queries.rel_weaver import MakeList

IndexQuery = BuiltinVars.IndexQuery
KeyQuery = BuiltinVars.KeyQuery