from .common_imports import *
from .core.model import ValueRef, Call, FuncOp, unwrap, wrap
from .core.sig import Signature
from .core.config import Config
from .ui.main import Storage, Context, MODES, FreeContexts
from .ui.funcs import op, Q

### testing stuff
from .storages.rel_impls.duckdb_impl import DuckDBRelStorage
from .tests.utils import *
from .storages.rels import serialize, deserialize

from mandala_lite.py_call_graphs.main import *
from mandala_lite.tests.sample_codebase.b import *


def test():
    g = DependencyGraph.from_code2flow(Path("./mandala_lite/tests/sample_codebase/"))
    deps, node_to_obj = g.get_dependencies(method_caller)
    return deps, node_to_obj
