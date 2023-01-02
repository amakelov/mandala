"""
Intended way for users to import mandala
"""
from .core.model import wrap
from .core.wrapping import unwrap
from .core.weaver import BuiltinQueries
from .core.config import Config
from .ui.main import Storage, Context, OnChange
from .ui.funcs import op, superop, Q
